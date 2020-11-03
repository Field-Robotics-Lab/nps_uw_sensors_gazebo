// #include <nps_uw_sensors_gazebo/sonar_calculation_cuda.h>

// __global__ 
// void NpsGazeboSonar::CudaTest(float *out, float *a, float *b, int n)
// {
//     for(int i = 0; i < n; i++){
//         out[i] = a[i] + b[i];
//     }
//     printf("CUDA Function running!!");
//     printf("CUDA Function running!!");
//     printf("CUDA Function running!!");
//     printf("CUDA Function running!!");
//     printf("CUDA Function running!!");
//     printf("CUDA Function running!!");
// }

// #include <nps_uw_sensors_gazebo/sonar_calculation_cuda.cuh>

// __device__ double data;

// __global__ void sonar_calculation_kernel(void) {

// 	// insert data to pass
// 	data = 422.146146;
// }

// namespace NpsGazeboSonar {
// 	void sonar_calculation(void)
// 	{
// 		sonar_calculation_kernel <<<1, 1>>> ();

// 		// Pass data
// 		typeof(data) answer;
// 		cudaMemcpyFromSymbol(&answer, data, sizeof(double), cudaMemcpyDeviceToHost);
// 		printf("answer: %f\n", answer);
// 	}
// }

#include <nps_uw_sensors_gazebo/sonar_calculation_cuda.cuh>

#include <math.h>
#include <assert.h>

// For complex numbers
#include <thrust/complex.h>
#include <cuComplex.h>

// For rand() function
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

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
  float camera_x = cos(-azimuth)*cos(elevation);
  float camera_y = sin(-azimuth)*cos(elevation);
  float camera_z = sin(elevation);
  float ray_normal[3] = {camera_x, camera_y, camera_z};
  
  // target normal with axes compensated to camera axes
//   float norm = sqrt(pow(normal[0],2) + pow(normal[1],2) + pow(normal[2],2));
//   float target_normal[3] = {normal[2]/norm, -normal[0]/norm, -normal[1]/norm};
  float target_normal[3] = {normal[2], -normal[0], -normal[1]};

  // dot product
  float dot_product = ray_normal[0]*target_normal[0]
  					   + ray_normal[1]*target_normal[1]
						 + ray_normal[2]*target_normal[2];
  if (dot_product < -1)
	  dot_product = -1;
  if (dot_product > 1)
	  dot_product = 1;

  return M_PI - acos(dot_product);
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
/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states)
{
	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
				blockIdx.x, /* the sequence number should be different for each core (unless you want all
							   cores to get the same sequence of numbers for some reason - use thread id! */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&states[blockIdx.x]);
}
__global__ void init2(unsigned int seed, curandState_t* states)
{
	curand_init(seed, blockIdx.x, 100, &states[blockIdx.x]);
}

///////////////////////////////////////////////////////////////////////////
// Sonar Claculation Function
__global__ void sonar_calculation(thrust::complex<float> *P_Beams,
									unsigned char* depth_image,
									unsigned char* normal_image,
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
									int nBeams, float sonarFreq, float df, float fmin,
									int nFreq, float bandwidth,
									float mu, float attenuation)
{

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height))
	{
		// Location of the image pixel
		const int depth_index = yIndex * depth_image_step + xIndex;
		const int normal_index = yIndex * normal_image_step + (3 * xIndex);
		const int rand_index = yIndex * rand_image_step + (2 * xIndex);

		// Input parameters for ray processing
		float ray_azimuthAngle = -(hFOV/2.0) + xIndex * hPixelSize + hPixelSize/2.0;
		float ray_elevationAngle = (vFOV/2.0) - yIndex * vPixelSize - vPixelSize/2.0;
		float distance = depth_image[depth_index] * 0.5f;
		float normal[3] = {normal_image[normal_index] * 0.003921569f,
							normal_image[normal_index + 1] * 0.003921569f,
							normal_image[normal_index + 2] * 0.003921569f};

		// Beam pattern
		float azimuthBeamPattern = pow(abs(unnormalized_sinc(M_PI * 0.884
						/ ray_azimuthAngleWidth * sin(ray_azimuthAngle))), 2);
		float elevationBeamPattern = pow(abs(unnormalized_sinc(M_PI * 0.884
						/ ray_elevationAngleWidth * sin(ray_elevationAngle))), 2);

		// incidence angle
		float incidence = compute_incidence(ray_azimuthAngle, ray_elevationAngle, normal);

		// ----- Point scattering model ------ //
		// generate a random number, (Gaussian noise)
		float xi_z = rand_image[rand_index] * 0.003921569f;  
		float xi_y = rand_image[rand_index + 1] * 0.003921569f;

		// Calculate amplitude
		thrust::complex<float> random_unit = thrust::complex<float>(xi_z, xi_y);
		thrust::complex<float> area = 
					thrust::complex<float>(sqrt(mu * pow(cos(incidence), 2) * pow(distance, 2)
					* ray_azimuthAngleWidth * ray_elevationAngleWidth), 0.0);
		thrust::complex<float> beamPattern = 
					thrust::complex<float>(azimuthBeamPattern * elevationBeamPattern, 0.0);
		thrust::complex<float> rms_scaler = 
					thrust::complex<float>(1.0 / sqrt(2.0), 0.0);
		thrust::complex<float> amplitude = random_unit * rms_scaler * area * beamPattern;
		
		// printf("1 %f\n", incidence);
		// // printf("beamPattern %f\n", beamPattern.real());
		// // printf("rms_scalaer %f\n", rms_scaler.real());
		// printf("amp %f\n", amplitude.real());

		// printf("1 %f\n", incidence);
		// printf("r0 %f\n", ray_normal[0]);
		// printf("r1 %f\n", ray_normal[1]);
		// printf("r2 %f\n", ray_normal[2]);
		// printf("t0 %f\n", target_normal[0]);
		// printf("t1 %f\n", target_normal[1]);
		// printf("t2 %f\n", target_normal[2]);
		// printf("%f\n", dot_product);	
		// asm("trap;");


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
			float S_f = 1e11 * exp(-pow(freq - sonarFreq, 2)
					* pow(M_PI, 2) / pow(bandwidth, 2));

			P_Beams[xIndex*nFreq + f] = P_Beams[xIndex*nFreq + f] + S_f * amplitude
						* exp(minusUnitImag * K * complexDistance * thrust::complex<float>(2.0, 0.0)) 
						/ pow(complexDistance, 2);
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
		fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
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
									int _nBeams, double _sonarFreq,
									double _fmax, double _fmin,
									double _bandwidth,
									double _mu,
									double _attenuation)
	{
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
		int nBeams = _nBeams;

		// ---------   Calculation parameters   --------- //
		double min_dist, max_dist;
		cv::minMaxLoc(depth_image, &min_dist, &max_dist);
		float max_distance = (float) max_dist;
		// Sampling periods
		float max_T = max_distance*2.0/soundSpeed;
		float delta_f = 1.0/max_T;
		const int nFreq = int(round(((float) _fmax  - (float) _fmin ) / delta_f));
		float df = ( (float) _fmax - (float) _fmin)/(nFreq-1);

		
		// ---------   Allocate GPU memory for image   --------- //
		// rand number generator 
		cv::Mat rand_image = cv::Mat(depth_image.rows, depth_image.cols, CV_8UC2);
		cv::randu(rand_image, cv::Scalar::all(0), cv::Scalar::all(255));
		//Calculate total number of bytes of input and output image
		const int depth_image_Bytes = depth_image.step * depth_image.rows;
		const int normal_image_Bytes = normal_image.step * normal_image.rows;
		const int rand_image_Bytes = rand_image.step * rand_image.rows;

		unsigned char *d_depth_image, *d_normal_image, *d_rand_image;

		//Allocate device memory
		SAFE_CALL(cudaMalloc<unsigned char>(&d_depth_image,depth_image_Bytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_normal_image,normal_image_Bytes),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_rand_image,rand_image_Bytes),"CUDA Malloc Failed");

		//Copy data from OpenCV input image to device memory
		SAFE_CALL(cudaMemcpy(d_depth_image,depth_image.ptr(),depth_image_Bytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaMemcpy(d_normal_image,normal_image.ptr(),normal_image_Bytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
		SAFE_CALL(cudaMemcpy(d_rand_image,rand_image.ptr(),rand_image_Bytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

		//Specify a reasonable block size
		const dim3 block(16,16);

		//Calculate grid size to cover the whole image
		const dim3 grid((normal_image.cols + block.x - 1)/block.x, (normal_image.rows + block.y - 1)/block.y);

		// Pixcel array
		thrust::complex<float> *P_Beams;
		thrust::complex<float> *d_P_Beams;
		const int P_Beams_Bytes = sizeof(thrust::complex<float>) * nBeams * nFreq;
		P_Beams = (thrust::complex<float>*)malloc(P_Beams_Bytes);
		SAFE_CALL(cudaMalloc<thrust::complex<float>>(&d_P_Beams,P_Beams_Bytes),"CUDA Malloc Failed");

		//Launch the color conversion kernel
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
										   nBeams, sonarFreq, df, fmin,
										   nFreq, bandwidth,
										   mu, attenuation);

		//Synchronize to check for any kernel launch errors
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

		//Copy back data from destination device meory to OpenCV output image
		SAFE_CALL(cudaMemcpy(P_Beams,d_P_Beams,P_Beams_Bytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

		// Assign values to Array for return
		CArray2D P_Beams_F(CArray(nFreq), nBeams);
		for (size_t col = 0; col < nBeams; col ++)
		{
			for (size_t f = 0; f < nFreq; f++)
			{	
				P_Beams_F[col][f] = Complex(P_Beams[col * nFreq + f].real(),
									 		P_Beams[col * nFreq + f].imag());
			}
		}

		// Free GPU memory
		cudaFree(d_depth_image);
		cudaFree(d_normal_image);
		cudaFree(d_P_Beams);

		return P_Beams_F;


		// FFT


	}
}
