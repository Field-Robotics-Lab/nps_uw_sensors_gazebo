#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <complex>
#include <valarray>

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>

namespace NpsGazeboSonar {

	typedef std::complex<double> Complex;
	typedef std::valarray<Complex> CArray;
	typedef std::valarray<CArray> CArray2D;

    /// \brief CUDA Device Check Function Wrapper
	void check_cuda_init_wrapper(void);

    /// \brief Sonar Claculation Function Wrapper
	// void sonar_calculation_wrapper(void);
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
										float *_window,
										float **_rayCorrector,
										float _rayCorrectorSum,
										float **_beamCorrector,
										float _beamCorrectorSum);

    /// \brief Incident Angle Calculation Function Wrapper
	void incident_angle_wrapper(float &_angle, float _azimuth,
								float _elevation, float *_normal);
}