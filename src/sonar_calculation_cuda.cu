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

#include <nps_uw_sensors_gazebo/sonar_calculation_cuda.cuh>

__global__ void test_kernel(void) {
}

namespace Wrapper {
	void wrapper(void)
	{
		test_kernel <<<1, 1>>> ();
		printf("Hello, world!");
	}
}