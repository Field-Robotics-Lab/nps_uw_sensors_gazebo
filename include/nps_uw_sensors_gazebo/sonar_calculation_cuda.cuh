// #ifndef __sonar_calculation_cuda_h__
// #define __sonar_calculation_cuda_h__

// #include <opencv2/core.hpp>
// #include <complex>
// #include <valarray>
// #include <sstream>
// #include <chrono>

// typedef std::complex<double> Complex;
// typedef std::valarray<Complex> CArray;
// typedef std::valarray<CArray> CArray2D;
  
// class NpsGazeboSonar
// {
//     /// \brief Parameters for sonar properties
//     // private: double sonarFreq;
//     // private: double bandwidth;
//     // private: double freqResolution;
//     // private: double soundSpeed;
//     // private: bool constMu;
//     // private: double absorption;
//     // private: double attenuation;
//     // private: double mu; // surface reflectivity
//     // private: double fmin;
//     // private: double fmax;
//     // private: double df;
//     // private: int sonarCalcWidthSkips;
//     // private: int sonarCalcHeightSkips;
//     // private: int nBeams;
//     // private: int ray_nAzimuthRays;
//     // private: int ray_nElevationRays;

//     public: 
//         __global__ void CudaTest(float *out, float *a, float *b, int n);
// };

// #endif

#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

#include <stdio.h>
#include <iostream>

namespace NpsGazeboSonar {
	void sonar_calculation(void);
}