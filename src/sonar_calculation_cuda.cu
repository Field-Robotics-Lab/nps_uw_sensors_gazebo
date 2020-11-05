#include <nps_uw_sensors_gazebo/sonar_calculation_cuda.cuh>

// #include <math.h>
#include <assert.h>

// For complex numbers
#include <thrust/complex.h>
#include <cuComplex.h>

// For rand() function
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

// For FFT
#include <cufft.h>
#include <cufftw.h>
#include <list>
#include <thrust/device_vector.h>

#include <chrono>

static inline void _safe_cuda_call(cudaError err, const char* msg,
									const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",
				msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

///////////////////////////////////////////////////////////////////////////
// Incident Angle Calculation Function
// incidence angle is target's normal angle accounting for the ray's azimuth
// and elevation
__device__ float compute_incidence(float azimuth, float elevation, float *normal)
{
  // ray normal from camera azimuth and elevation
  float camera_x = cosf(-azimuth)*cosf(elevation);
  float camera_y = sinf(-azimuth)*cosf(elevation);
  float camera_z = sinf(elevation);
  float ray_normal[3] = {camera_x, camera_y, camera_z};

  // target normal with axes compensated to camera axes
  //   float norm = sqrt(pow(normal[0],2) + pow(normal[1],2) + pow(normal[2],2));
  //   float target_normal[3] = {normal[2]/norm, -normal[0]/norm, -normal[1]/norm};
  float target_normal[3] = {normal[2], -normal[0], -normal[1]};

  // dot product
  float dot_product = ray_normal[0]*target_normal[0]
  					   + ray_normal[1]*target_normal[1]
						 + ray_normal[2]*target_normal[2];

  if (dot_product < -1.0) dot_product = -1.0;
  if (dot_product > 1.0) dot_product = 1.0;

//   printf("TEST    %f \n", dot_product);

  return M_PI - acosf(dot_product);
}

///////////////////////////////////////////////////////////////////////////
__device__ float unnormalized_sinc(float t)
{
	if (abs(t) < 1E-8)
		return 1.0;
	else
		return sin(t)/t;
}

///////////////////////////////////////////////////////////////////////////
// Sonar Claculation Function
__global__ void sonar_calculation(thrust::complex<float> *P_Beams,
									float* depth_image,
									float* normal_image,
									int width,
									int height,
									int depth_image_step,
									int normal_image_step,
									unsigned char* rand_image,
									int rand_image_step,
									float hPixelSize,
									float vPixelSize,
									float hFOV,
									float vFOV,
									float beam_elevationAngleWidth,
									float beam_azimuthAngleWidth,
									float ray_elevationAngleWidth,
									float ray_azimuthAngleWidth,
									float soundSpeed,
									int nBeams, int nRays,
									int beamSkips, int raySkips,
									float sonarFreq, float df, float fmin,
									int nFreq, float bandwidth,
									float mu, float attenuation)
{
	//2D Index of current thread
	const int beam = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((beam<width) && (ray<height) && (beam % beamSkips == 0) && (ray % raySkips == 0))
	{
		// Location of the image pixel
		const int depth_index = ray * depth_image_step/sizeof(float) + beam;
		const int normal_index = ray * normal_image_step/sizeof(float) + (3 * beam);
		const int rand_index = ray * rand_image_step/sizeof(float) + (2 * beam);
		// Input parameters for ray processing
		float ray_azimuthAngle = -(hFOV/2.0) + beam * hPixelSize + hPixelSize/2.0;
		float ray_elevationAngle = (vFOV/2.0) - ray * vPixelSize - vPixelSize/2.0;
		float distance = depth_image[depth_index] * 0.5f;
		float normal[3] = {normal_image[normal_index],
							normal_image[normal_index + 1],
							normal_image[normal_index + 2]};

		// Beam pattern
		float azimuthBeamPattern = pow(abs(unnormalized_sinc(M_PI * 0.884
						/ ray_azimuthAngleWidth * sin(ray_azimuthAngle))), 2);
		float elevationBeamPattern = pow(abs(unnormalized_sinc(M_PI * 0.884
						/ ray_elevationAngleWidth * sin(ray_elevationAngle))), 2);

		// incidence angle
		float incidence = compute_incidence(ray_azimuthAngle, ray_elevationAngle, normal);

		// ----- Point scattering model ------ //
		// generate a random number, (Gaussian noise)
		// TODO : it's acting like a fixed value in a frame. I should be random within a frame
		float xi_z = rand_image[rand_index] * 0.003921569f;
		float xi_y = rand_image[rand_index + 1] * 0.003921569f;
		// // Calculate amplitude
		thrust::complex<float> random_unit = thrust::complex<float>(xi_z, xi_y);
		thrust::complex<float> area =
					thrust::complex<float>(sqrt(mu * pow(cos(incidence), 2) * pow(distance, 2)
					* ray_azimuthAngleWidth * ray_elevationAngleWidth), 0.0);
		thrust::complex<float> beamPattern =
					thrust::complex<float>(azimuthBeamPattern * elevationBeamPattern, 0.0);
		thrust::complex<float> rms_scaler =
					thrust::complex<float>(1.0 / sqrt(2.0), 0.0);
		thrust::complex<float> amplitude = random_unit * rms_scaler * area * beamPattern;

		// Summation of Echo returned from a signal (frequency domain)
		for (size_t f = 0; f < nFreq; f++)  // frequency loop from fmin to fmax
		{
			float freq = fmin + df*f;
			float kw = 2.0*M_PI*freq/soundSpeed;   // wave vector
			thrust::complex<float> K =
					thrust::complex<float>(kw, attenuation);  // attenuation constant K1f
			thrust::complex<float> minusUnitImag = thrust::complex<float>(0.0, -1.0);
			thrust::complex<float> complexDistance = thrust::complex<float>(distance, 0.0);
			// Transmit spectrum, frequency domain
			thrust::complex<float> S_f =
					thrust::complex<float>(1e11 * exp(-pow(freq - sonarFreq, 2)
											* pow(M_PI, 2) / pow(bandwidth, 2)), 0.0);

			P_Beams[beam * nFreq * (int)(nRays/raySkips) + ray/raySkips * nFreq + f] =
						S_f * amplitude * exp(minusUnitImag * K * complexDistance
							* thrust::complex<float>(2.0, 0.0)) / pow(complexDistance, 2);
		}
	}
	// Stopper for debugging
	// asm("trap;");
}

///////////////////////////////////////////////////////////////////////////
namespace NpsGazeboSonar {

	// CUDA Device Checker Wrapper
	void check_cuda_init_wrapper(void)
	{
		// Check CUDA device
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error!=cudaSuccess)
		{
		fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error));
		exit(-1);
		}
	}

	// Sonar Claculation Function Wrapper
	CArray2D sonar_calculation_wrapper( const cv::Mat& depth_image,
									const cv::Mat& normal_image,
									double _hPixelSize,
									double _vPixelSize,
									double _hFOV,
									double _vFOV,
									double _beam_elevationAngleWidth,
									double _beam_azimuthAngleWidth,
									double _ray_elevationAngleWidth,
									double _ray_azimuthAngleWidth,
									double _soundSpeed,
									int _nBeams, int _nRays,
									int _beamSkips, int _raySkips,
									double _sonarFreq,
									double _fmax, double _fmin,
									double _bandwidth,
									double _mu,
									double _attenuation)
	{
		// auto start = std::chrono::high_resolution_clock::now();

		// ----  Allocation of properties parameters  ---- //
		float hPixelSize = (float) _hPixelSize;
		float vPixelSize = (float) _vPixelSize;
		float hFOV = (float) _hFOV;
		float vFOV = (float) _vFOV;
		float beam_elevationAngleWidth = (float) _beam_elevationAngleWidth;  // [radians]
		float beam_azimuthAngleWidth = (float) _beam_azimuthAngleWidth;      // [radians]
		float ray_elevationAngleWidth = (float) _ray_elevationAngleWidth;
		float ray_azimuthAngleWidth = (float) _ray_azimuthAngleWidth;
		float soundSpeed = (float) _soundSpeed;
		float sonarFreq = (float) _sonarFreq;
		float bandwidth = (float) _bandwidth;
		float fmin = (float) _fmin;
		float mu = (float) _mu;
		float attenuation = (float) _attenuation;
		int nBeams = _nBeams; int nRays = _nRays;
		int beamSkips = _beamSkips; int raySkips = _raySkips;

		// ---------   Calculation parameters   --------- //
		double min_dist, max_dist;
		cv::minMaxLoc(depth_image, &min_dist, &max_dist);
		float max_distance = (float) max_dist;
		// Sampling periods
		float max_T = max_distance*2.0/soundSpeed;
		float delta_f = 1.0/max_T;
		const int nFreq = int(round(((float) _fmax  - (float) _fmin ) / delta_f));
		float df = ( (float) _fmax - (float) _fmin)/(nFreq-1);
		ray_azimuthAngleWidth = ray_azimuthAngleWidth * (beamSkips * 1.0f);
		ray_elevationAngleWidth = ray_elevationAngleWidth * (raySkips * 1.0f);

		// ---------   Allocate GPU memory for image   --------- //
		// rand number generator
		cv::Mat rand_image = cv::Mat(depth_image.rows, depth_image.cols, CV_8UC2);
		cv::randu(rand_image, cv::Scalar::all(0), cv::Scalar::all(255));
		//Calculate total number of bytes of input and output image
		const int depth_image_Bytes = depth_image.step * depth_image.rows;
		const int normal_image_Bytes = normal_image.step * normal_image.rows;
		const int rand_image_Bytes = rand_image.step * rand_image.rows;

		float *d_depth_image, *d_normal_image;
		unsigned char *d_rand_image;

		//Allocate device memory
		SAFE_CALL(cudaMalloc<float>(&d_depth_image,depth_image_Bytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<float>(&d_normal_image,normal_image_Bytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_rand_image,rand_image_Bytes),"CUDA Malloc Failed");

		//Copy data from OpenCV input image to device memory
		SAFE_CALL(cudaMemcpy(d_depth_image,depth_image.ptr(),depth_image_Bytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaMemcpy(d_normal_image,normal_image.ptr(),normal_image_Bytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaMemcpy(d_rand_image,rand_image.ptr(),rand_image_Bytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

		//Specify a reasonable block size
		const dim3 block(16, 16);

		// //Calculate grid size to cover the whole image
		const dim3 grid((depth_image.cols + block.x - 1)/block.x, 
						(depth_image.rows + block.y - 1)/block.y);

		// Pixcel array
		thrust::complex<float> *P_Beams;
		thrust::complex<float> *d_P_Beams;
		const int P_Beams_N = (int)(nBeams/beamSkips) * (int)(nRays/raySkips) * nFreq;
		const int P_Beams_Bytes = sizeof(thrust::complex<float>) * P_Beams_N;
		P_Beams = (thrust::complex<float>*)malloc(P_Beams_Bytes);
		SAFE_CALL(cudaMalloc<thrust::complex<float>>(&d_P_Beams,P_Beams_Bytes),"CUDA Malloc Failed");

		//Launch the beamor conversion kernel
		sonar_calculation<<<grid,block>>>(d_P_Beams,
										   d_depth_image,
										   d_normal_image,
										   normal_image.cols,
										   normal_image.rows,
										   depth_image.step,
										   normal_image.step,
										   d_rand_image,
										   rand_image.step,
										   hPixelSize,
										   vPixelSize,
										   hFOV,
										   vFOV,
										   beam_elevationAngleWidth,
										   beam_azimuthAngleWidth,
										   ray_elevationAngleWidth,
										   ray_azimuthAngleWidth,
										   soundSpeed,
										   nBeams, nRays,
										   beamSkips, raySkips,
										   sonarFreq, df, fmin,
										   nFreq, bandwidth,
										   mu, attenuation);

		//Synchronize to check for any kernel launch errors
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

		//Copy back data from destination device meory to OpenCV output image
		SAFE_CALL(cudaMemcpy(P_Beams,d_P_Beams,P_Beams_Bytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

		// Free GPU memory
		cudaFree(d_depth_image);
		cudaFree(d_normal_image);
		cudaFree(d_P_Beams);

		// // For calc time measure
		// auto stop = std::chrono::high_resolution_clock::now();
		// auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		// printf("GPU Sonar Calc Time %lld/100 [s]\n",static_cast<long long int>(duration.count()/10000));
		// start = std::chrono::high_resolution_clock::now();

		// Preallocate an array for return
		CArray2D P_Beams_F(CArray(nFreq), nBeams);

		// Ray summation
		for (size_t beam = 0; beam < nBeams; beam += beamSkips)
		{
			for (size_t f = 0; f < nFreq; f++)
			{
				float P_Beam_real = 0.0;
				float P_Beam_imag = 0.0;
				for (size_t ray = 0; ray < nRays; ray += raySkips)
				{
					P_Beam_real += P_Beams[beam * nFreq * (int)(nRays/raySkips) + ray/raySkips * nFreq + f].real();
					P_Beam_imag += P_Beams[beam * nFreq * (int)(nRays/raySkips) + ray/raySkips * nFreq + f].imag();
				}
				P_Beams_F[beam][f] = Complex(P_Beam_real, P_Beam_imag);
			}
		}

		// For calc time measure
		// stop = std::chrono::high_resolution_clock::now();
		// duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		// printf("GPU Sonar Summation %lld/100 [s]\n",static_cast<long long int>(duration.count()/10000));
		// start = std::chrono::high_resolution_clock::now();

		// ========== FFT ======== //
		const int DATASIZE = pow(2, ceil(log(nFreq)/log(2)));
		const int BATCH = nBeams;
		// --- Host side input data allocation and initialization
		cufftComplex *hostInputData = (cufftComplex*)malloc(DATASIZE*BATCH*sizeof(cufftComplex));
		for (int beam = 0; beam < BATCH; beam++)
		{
			for (int f = 0; f < DATASIZE; f++)
			{
				if (f < nFreq)
					hostInputData[beam*DATASIZE + f] = make_cuComplex(P_Beams_F[beam][f].real(),
																	  P_Beams_F[beam][f].imag());
				else
					hostInputData[beam*DATASIZE + f] = (make_cuComplex(0.f, 0.f));  // zero padding
			}
		}

		// --- Device side input data allocation and initialization
		cufftComplex *deviceInputData;
		SAFE_CALL(cudaMalloc((void**)&deviceInputData, DATASIZE * BATCH * sizeof(cufftComplex)), "FFT CUDA Malloc Failed");
		SAFE_CALL(cudaMemcpy(deviceInputData, hostInputData, DATASIZE * BATCH * sizeof(cufftComplex), cudaMemcpyHostToDevice), "FFT CUDA Memcopy Failed");

		// --- Host side output data allocation
		cufftComplex *hostOutputData = (cufftComplex*)malloc(DATASIZE* BATCH * sizeof(cufftComplex));

		// --- Device side output data allocation
		cufftComplex *deviceOutputData; cudaMalloc((void**)&deviceOutputData, DATASIZE * BATCH * sizeof(cufftComplex));

		// --- Batched 1D FFTs
		cufftHandle handle;
		int rank = 1;                           // --- 1D FFTs
		int n[] = { DATASIZE };                 // --- Size of the Fourier transform
		int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
		int idist = DATASIZE, odist = DATASIZE; // --- Distance between batches
		int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
		int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
		int batch = BATCH;                      // --- Number of batched executions
		cufftPlanMany(&handle, rank, n,
						inembed, istride, idist,
						onembed, ostride, odist, CUFFT_C2C, batch);

		cufftExecC2C(handle,  deviceInputData, deviceOutputData, CUFFT_INVERSE);

		// --- Device->Host copy of the results
		SAFE_CALL(cudaMemcpy(hostOutputData, deviceOutputData, DATASIZE * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost),"FFT CUDA Memcopy Failed");

		cufftDestroy(handle);
		cudaFree(deviceOutputData);
		cudaFree(deviceInputData);

		for (int beam = 0; beam < BATCH; beam++)
		{
			for (int f = 0; f < nFreq; f++)
			{
				P_Beams_F[beam][f] = Complex(hostOutputData[beam*DATASIZE + f].x/DATASIZE,
											hostOutputData[beam*DATASIZE + f].y/DATASIZE);
			}
		}

		// For calc time measure
		// stop = std::chrono::high_resolution_clock::now();
		// duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		// printf("GPU FFT Calc Time %lld/100 [s]\n",static_cast<long long int>(duration.count()/10000));

		return P_Beams_F;
	}
}
