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

#define N 100000
#define MAX_ERR 1e-5

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

namespace NpsGazeboSonar {
	void sonar_calculation(void)
	{
		// Check CUDA device
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error!=cudaSuccess)
		{
		fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
		exit(-1);
		}

		float *a, *b, *out;
		float *d_a, *d_b, *d_out;

		// Allocate host memory
		a   = (float*)malloc(sizeof(float) * N);
		b   = (float*)malloc(sizeof(float) * N);
		out = (float*)malloc(sizeof(float) * N);

		// Initialize host array
		for(int i = 0; i < N; i++){
			a[i] = 1.0f;
			b[i] = 2.0f;
		}

		// Allocate GPU device memory
		cudaMalloc((void**)&d_a, sizeof(float) * N);
		cudaMalloc((void**)&d_b, sizeof(float) * N);
		cudaMalloc((void**)&d_out, sizeof(float) * N);

		// Copy data from host to GPU Device
		cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
	
		// Main function
		vector_add<<<1,1>>>(d_out, d_a, d_b, N);
    
		// Transfer data back to host memory
		cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

		// Verification
		for(int i = 0; i < N; i++){
			printf("a = %f\n", a[i]);
			printf("b = %f\n", b[i]);
			printf("out = %f\n", out[i]);
			printf("error = %f\n", fabs(out[i] - a[i] - b[i]));
			assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
		}
		printf("out[0] = %f\n", out[0]);
		printf("PASSED\n");

		// Free GPU memory
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_out);

		// Free host memory
		free(a); 
		free(b); 
		free(out);
	}
}
