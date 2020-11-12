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
#include <thrust/device_vector.h>
#include <list>

#include <chrono>

#define BLOCK_SIZE 512

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
__device__ __host__ float unnormalized_sinc(float t)
{
	if (abs(t) < 1E-8)
		return 1.0;
	else
		return sin(t)/t;
}

///////////////////////////////////////////////////////////////////////////
__global__ void total(thrust::complex<float> *input, thrust::complex<float> *output, int len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ thrust::complex<float> partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = thrust::complex<float>(0.0, 0.0);
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}

__global__ void sum(thrust::complex<float>* input)
{
	const int tid = threadIdx.x;

	auto step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
		}

		step_size <<= 1;
		number_of_threads >>= 1;
	}
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
									float* rand_image,
									int rand_image_step,
									float hPixelSize,
									float vPixelSize,
									float hFOV,
									float vFOV,
									float beam_azimuthAngleWidth,
									float beam_elevationAngleWidth,
									float ray_azimuthAngleWidth,
									float ray_elevationAngleWidth,
									float soundSpeed,
									float sourceTerm,
									int nBeams, int nRays,
									int beamSkips, int raySkips,
									float sonarFreq, float delta_f,
									int nFreq, float bandwidth,
									float mu_sqrt, float attenuation,
									float area_scaler)
{
	// 2D Index of current thread
	const int beam = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((beam<width) && (ray<height) &&
			 (beam % beamSkips == 0) && (ray % raySkips == 0))
	{
		// Location of the image pixel
		const int depth_index = ray * depth_image_step/sizeof(float) + beam;
		const int normal_index = ray * normal_image_step/sizeof(float) + (3 * beam);
		const int rand_index = ray * rand_image_step/sizeof(float) + (2 * beam);
		// Input parameters for ray processing
		float ray_azimuthAngle = -(hFOV/2.0) + beam * hPixelSize + hPixelSize/2.0;
		float ray_elevationAngle = (vFOV/2.0) - ray * vPixelSize - vPixelSize/2.0;
		float distance = depth_image[depth_index] * 1.0f;
		float normal[3] = {normal_image[normal_index],
							normal_image[normal_index + 1],
							normal_image[normal_index + 2]};

		// Beam pattern
		// float azimuthBeamPattern = abs(unnormalized_sinc(M_PI * 0.884
		// 				/ ray_azimuthAngleWidth * sin(ray_azimuthAngle)));
		// only one column of rays for each beam at beam center
		float azimuthBeamPattern = 1.0;
		float elevationBeamPattern = unnormalized_sinc(M_PI * 0.884
						/ vFOV * sin(ray_elevationAngle));
		// incidence angle
		float incidence = compute_incidence(ray_azimuthAngle, ray_elevationAngle, normal);

		// ----- Point scattering model ------ //
		// Gaussian noise generated using opencv RNG
		float xi_z = rand_image[rand_index];
		float xi_y = rand_image[rand_index + 1];

		// Calculate amplitude
		thrust::complex<float> randomAmps = thrust::complex<float>(xi_z/sqrt(2.0), xi_y/sqrt(2.0));
		thrust::complex<float> lambert_sqrt =
					thrust::complex<float>(mu_sqrt*cos(incidence), 0.0);
		thrust::complex<float> beamPattern =
					thrust::complex<float>(azimuthBeamPattern * elevationBeamPattern, 0.0);
		thrust::complex<float> targetArea_sqrt = thrust::complex<float>(sqrt(distance * area_scaler), 0.0);
		thrust::complex<float> propagationTerm =
					thrust::complex<float>(1.0/pow(distance, 2.0) * exp(-2.0 * attenuation * distance), 0.0);

		thrust::complex<float> amplitude = randomAmps * thrust::complex<float>(sourceTerm, 0.0)
					* propagationTerm * beamPattern * lambert_sqrt * targetArea_sqrt;

		// Summation of Echo returned from a signal (frequency domain)
		for (size_t f = 0; f < nFreq; f++)
		{
			float freq;
			if (nFreq % 2 == 0)
				freq = delta_f * (-nFreq/2 + f + 1);
			else
				freq = delta_f * (-(nFreq-1)/2 + f + 1);
			float kw = 2.0*M_PI*freq/soundSpeed;  // wave vector
			thrust::complex<float> unitImag = thrust::complex<float>(0.0, 1.0);
			thrust::complex<float> complexDistance = thrust::complex<float>(0.0, distance);
			// Transmit spectrum, frequency domain
			P_Beams[beam * nFreq * (int)(nRays/raySkips) + (int)(ray/raySkips) * nFreq + f] =
					exp(thrust::complex<float>(0.0, 2*distance*kw)) * amplitude;
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
									const cv::Mat& rand_image,
									double _hPixelSize,
									double _vPixelSize,
									double _hFOV,
									double _vFOV,
									double _beam_azimuthAngleWidth,
									double _beam_elevationAngleWidth,
									double _ray_azimuthAngleWidth,
									double _ray_elevationAngleWidth,
									double _soundSpeed,
									double _maxDistance,
									double _sourceLevel,
									int _nBeams, int _nRays,
									int _beamSkips, int _raySkips,
									double _sonarFreq,
									double _bandwidth,
									int _nFreq,
									double _mu,
									double _attenuation,
									float *window,
									float **rayCorrector,
									float rayCorrectorSum,
									float **beamCorrector,
									float beamCorrectorSum)
	{
		auto start = std::chrono::high_resolution_clock::now();

		// ----  Allocation of properties parameters  ---- //
		const float hPixelSize = (float) _hPixelSize;
		const float vPixelSize = (float) _vPixelSize;
		const float hFOV = (float) _hFOV;
		const float vFOV = (float) _vFOV;
		const float beam_elevationAngleWidth = (float) _beam_elevationAngleWidth;
		const float beam_azimuthAngleWidth = (float) _beam_azimuthAngleWidth;
		const float ray_elevationAngleWidth = (float) _ray_elevationAngleWidth;
		const float ray_azimuthAngleWidth = (float) _ray_azimuthAngleWidth;
		const float soundSpeed = (float) _soundSpeed;
		const float maxDistance = (float) _maxDistance;
		const float sonarFreq = (float) _sonarFreq;
		const float bandwidth = (float) _bandwidth;
		const float mu = (float) _mu;
		const float attenuation = (float) _attenuation;
		const int nBeams = _nBeams; const int nRays = _nRays;
		const int nFreq = _nFreq;
		const int beamSkips = _beamSkips; const int raySkips = _raySkips;

		// ---------   Calculation parameters   --------- //
		// double min_dist, max_dist;
		// cv::minMaxLoc(depth_image, &min_dist, &max_dist);
		// float max_distance = (float) max_dist;
		const float max_distance = maxDistance;
		// Signal
		const float max_T = max_distance*2.0/soundSpeed;
		const float delta_f = 1.0/max_T;
		const float delta_t = 1.0/bandwidth;
		// const int nFreq = ceil(bandwidth/delta_f);
		// Precalculation
		const float mu_sqrt = sqrt(mu);
		const float area_scaler = ray_azimuthAngleWidth * ray_elevationAngleWidth;
		const float sourceLevel = (float) _sourceLevel;;  // db re 1 muPa;
		const float pref = 1e-6;  // 1 micro pascal (muPa);
		const float sourceTerm = sqrt(pow(10,(sourceLevel/10)))*pref;  // source term

		// ---------   Allocate GPU memory for image   --------- //
		//Calculate total number of bytes of input and output image
		const int depth_image_Bytes = depth_image.step * depth_image.rows;
		const int normal_image_Bytes = normal_image.step * normal_image.rows;
		const int rand_image_Bytes = rand_image.step * rand_image.rows;

		float *d_depth_image, *d_normal_image, *d_rand_image;

		//Allocate device memory
		SAFE_CALL(cudaMalloc<float>(&d_depth_image,depth_image_Bytes),
									"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<float>(&d_normal_image,normal_image_Bytes),
									"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<float>(&d_rand_image,rand_image_Bytes),
									"CUDA Malloc Failed");

		//Copy data from OpenCV input image to device memory
		SAFE_CALL(cudaMemcpy(
			d_depth_image,depth_image.ptr(),depth_image_Bytes,
			cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaMemcpy(
			d_normal_image,normal_image.ptr(),normal_image_Bytes,
			cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaMemcpy(
			d_rand_image,rand_image.ptr(),rand_image_Bytes,
			cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

		//Specify a reasonable block size
		const dim3 block(16, 16);

		// //Calculate grid size to cover the whole image
		const dim3 grid((depth_image.cols + block.x - 1)/block.x,
						(depth_image.rows + block.y - 1)/block.y);

		// Pixcel array
		thrust::complex<float> *P_Beams;
		thrust::complex<float> *d_P_Beams;
		const int P_Beams_N = (int)(nBeams/beamSkips) * (int)(nRays/raySkips) * (nFreq + 1);
		const int P_Beams_Bytes = sizeof(thrust::complex<float>) * P_Beams_N;
		P_Beams = (thrust::complex<float>*)malloc(P_Beams_Bytes);
		SAFE_CALL(cudaMalloc<thrust::complex<float>>(&d_P_Beams,P_Beams_Bytes),
									"CUDA Malloc Failed");

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
										   beam_azimuthAngleWidth,
										   beam_elevationAngleWidth,
										   ray_azimuthAngleWidth,
										   ray_elevationAngleWidth,
										   soundSpeed,
										   sourceTerm,
										   nBeams, nRays,
										   beamSkips, raySkips,
										   sonarFreq, delta_f,
										   nFreq, bandwidth,
										   mu_sqrt, attenuation,
										   area_scaler);

		//Synchronize to check for any kernel launch errors
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

		//Copy back data from destination device meory to OpenCV output image
		SAFE_CALL(cudaMemcpy(P_Beams,d_P_Beams,P_Beams_Bytes,
			cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

		// Free GPU memory
		cudaFree(d_depth_image);
		cudaFree(d_normal_image);
		cudaFree(d_rand_image);
		cudaFree(d_P_Beams);

		// For calc time measure
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		printf("GPU Sonar Computation Time %lld/100 [s]\n",static_cast<long long int>(duration.count()/10000));
		start = std::chrono::high_resolution_clock::now();

		// Preallocate an array for return
		CArray2D P_Beams_F(CArray(nFreq), nBeams);
		CArray2D P_Beams_F_Corrected(CArray(nFreq), nBeams);

		// // Array allocation for beams
		// thrust::complex<float> *P_Kernel;
		// thrust::complex<float> *d_P_Kernel;
		// const int P_Kernel_N = (int)(nRays/raySkips);
		// const int P_Kernel_Bytes = sizeof(thrust::complex<float>) * P_Kernel_N;
		// P_Kernel = (thrust::complex<float>*)malloc(P_Kernel_Bytes);
		// SAFE_CALL(cudaMalloc<thrust::complex<float>>(&d_P_Kernel,P_Kernel_Bytes),"CUDA Malloc Failed");

		// // GPU summation
        // for (size_t beam = 0; beam < nBeams; beam += beamSkips)
        // {
        //     for (size_t f = 0; f < nFreq; f++)
        //     {
		// 		// Allocate ray data
		// 		for (size_t ray = 0; ray < nRays; ray += raySkips)
		// 		{
		// 			P_Kernel[ray] =
		// 				thrust::complex<float>(P_Beams[beam * nFreq * (int)(nRays/raySkips)
		// 									    + (int)(ray/raySkips) * nFreq + f].real(),
		// 								       P_Beams[beam * nFreq * (int)(nRays/raySkips)
		// 									    + (int)(ray/raySkips) * nFreq + f].imag());
		// 		}
		// 		SAFE_CALL(cudaMemcpy(d_P_Kernel,P_Kernel,P_Kernel_Bytes,
		// 			cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

		// 		sum<<<1, (int)(nRays/raySkips)>>>(d_P_Kernel);

		// 		//Synchronize to check for any kernel launch errors
		// 		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

		// 		//Copy back data from destination device meory
		// 		thrust::complex<float> result;
		// 		SAFE_CALL(cudaMemcpy(&result,d_P_Kernel,sizeof(thrust::complex<float>),
		// 			cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

		// 		P_Beams_F[beam][f] = Complex(result.real() * window[f],
		// 									 result.imag() * window[f]);
		// 	}
		// }

		// // Free GPU memory
		// cudaFree(d_P_Kernel);

        // // CPU summation
        // for (size_t beam = 0; beam < nBeams; beam += beamSkips)
        // {
		// 	// Frequency loop
        //     for (size_t f = 0; f < nFreq; f++)
        //     {
        //         float P_Beam_real = 0.0;
        //         float P_Beam_imag = 0.0;
        //         for (size_t ray = 0; ray < nRays; ray += raySkips)
        //         {
        //             P_Beam_real += P_Beams[beam * nFreq * (int)(nRays/raySkips)
        //                              + (int)(ray/raySkips) * nFreq + f].real();
        //             P_Beam_imag += P_Beams[beam * nFreq * (int)(nRays/raySkips)
        //                              + (int)(ray/raySkips) * nFreq + f].imag();
        //         }
		// 		P_Beams_F[beam][f] = Complex(P_Beam_real * window[f],
        //                     				 P_Beam_imag * window[f]);
        //     }
		// }

		// // For calc time measure
		// stop = std::chrono::high_resolution_clock::now();
		// duration = std::chrono::duration_cast
		// 		<std::chrono::microseconds>(stop - start);
		// printf("GPU Sonar Summation %lld/100 [s]\n",
		// 		static_cast<long long int>(duration.count()/10000));
		// start = std::chrono::high_resolution_clock::now();


		// Ray culling correction and summation
		// Corrector and correctorSum is precalculated at parent cpp
        for (size_t beam = 0; beam < nBeams; beam += beamSkips)
        {
			// Frequency loop
            for (size_t f = 0; f < nFreq; f++)
            {
                float P_Beam_real = 0.0;
				float P_Beam_imag = 0.0;
                for (size_t ray = 0; ray < nRays; ray += raySkips)
                {
					for (size_t ray_other = 0; ray_other < nRays; ray_other += raySkips)
					{
						P_Beam_real += P_Beams[beam * nFreq * (int)(nRays/raySkips)
										 + (int)(ray_other/raySkips) * nFreq + f].real()
										 * rayCorrector[ray][ray_other];
						P_Beam_imag += P_Beams[beam * nFreq * (int)(nRays/raySkips)
										 + (int)(ray_other/raySkips) * nFreq + f].imag()
										 * rayCorrector[ray][ray_other];
					}
                }
				P_Beams_F[beam][f] = Complex(P_Beam_real * window[f] / rayCorrectorSum,
                            				 P_Beam_imag * window[f] / rayCorrectorSum);
            }
		}

		// Beam culling correction
		// Corrector and correctorSum is precalculated at parent cpp
        for (size_t beam = 0; beam < nBeams; beam += beamSkips)
        {
			for (size_t f = 0; f < nFreq; f++)
			{
				Complex corrected = Complex(0.0, 0.0);
				for (size_t beam_other = 0; beam_other < nBeams; beam_other += beamSkips)
				{
					corrected += P_Beams_F[beam_other][f]
								 * Complex(beamCorrector[beam][beam_other],
									       beamCorrector[beam][beam_other]);
				}
				P_Beams_F_Corrected[beam][f] =
						corrected/Complex(beamCorrectorSum, beamCorrectorSum);
			}
		}
		// return
		P_Beams_F = P_Beams_F_Corrected;

		// For calc time measure
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast
				<std::chrono::microseconds>(stop - start);
		printf("CPU Sonar Sum - Correction %lld/100 [s]\n",
				static_cast<long long int>(duration.count()/10000));
		start = std::chrono::high_resolution_clock::now();

		// ========== FFT ======== //
		// const int DATASIZE = pow(2, ceil(log(nFreq)/log(2)));
		const int DATASIZE = nFreq;
		const int BATCH = nBeams;
		// --- Host side input data allocation and initialization
		cufftComplex *hostInputData = (cufftComplex*)malloc(
								DATASIZE*BATCH*sizeof(cufftComplex));
		for (int beam = 0; beam < BATCH; beam++)
		{
			for (int f = 0; f < DATASIZE; f++)
			{
				if (f < nFreq)
					hostInputData[beam*DATASIZE + f] =
							make_cuComplex(P_Beams_F[beam][f].real(),
										   P_Beams_F[beam][f].imag());
				else
					hostInputData[beam*DATASIZE + f] =
							(make_cuComplex(0.f, 0.f));  // zero padding
			}
		}

		// --- Device side input data allocation and initialization
		cufftComplex *deviceInputData;
		SAFE_CALL(cudaMalloc((void**)&deviceInputData,
							DATASIZE * BATCH * sizeof(cufftComplex)),
							"FFT CUDA Malloc Failed");
		SAFE_CALL(cudaMemcpy(deviceInputData, hostInputData,
							DATASIZE * BATCH * sizeof(cufftComplex),
							cudaMemcpyHostToDevice), "FFT CUDA Memcopy Failed");

		// --- Host side output data allocation
		cufftComplex *hostOutputData =
				(cufftComplex*)malloc(DATASIZE* BATCH * sizeof(cufftComplex));

		// --- Device side output data allocation
		cufftComplex *deviceOutputData;
		cudaMalloc((void**)&deviceOutputData,
				DATASIZE * BATCH * sizeof(cufftComplex));

		// --- Batched 1D FFTs
		cufftHandle handle;
		int rank = 1;            // --- 1D FFTs
		int n[] = { DATASIZE };  // --- Size of the Fourier transform
		// --- Distance between two successive input/output elements
		int istride = 1, ostride = 1;
		int idist = DATASIZE, odist = DATASIZE; // --- Distance between batches
		// --- Input/Output size with pitch (ignored for 1D transforms)
		int inembed[] = { 0 };
		int onembed[] = { 0 };
		int batch = BATCH;       // --- Number of batched executions
		cufftPlanMany(&handle, rank, n,
						inembed, istride, idist,
						onembed, ostride, odist, CUFFT_C2C, batch);

		cufftExecC2C(handle,  deviceInputData, deviceOutputData, CUFFT_FORWARD);

		// --- Device->Host copy of the results
		SAFE_CALL(cudaMemcpy(hostOutputData, deviceOutputData,
						DATASIZE * BATCH * sizeof(cufftComplex),
						cudaMemcpyDeviceToHost),"FFT CUDA Memcopy Failed");

		cufftDestroy(handle);
		cudaFree(deviceOutputData);
		cudaFree(deviceInputData);

		for (int beam = 0; beam < BATCH; beam++)
		{
			for (int f = 0; f < nFreq; f++)
			{
				P_Beams_F[beam][f] =
					Complex(hostOutputData[beam*DATASIZE + f].x * delta_f,
							hostOutputData[beam*DATASIZE + f].y * delta_f);
			}
		}

		// For calc time measure
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast
							<std::chrono::microseconds>(stop - start);
		printf("GPU FFT Calc Time %lld/100 [s]\n",
							static_cast<long long int>(duration.count()/10000));

		return P_Beams_F;
	}
}
