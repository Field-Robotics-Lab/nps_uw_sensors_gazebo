/*
 * This file was modified from the original version within Gazebo:
 *
 * Copyright (C) 2014 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Modifications:
 *
 * Copyright 2018 Nils Bore (nbore@kth.se)
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef GAZEBO_ROS_IMAGE_SONAR_HH
#define GAZEBO_ROS_IMAGE_SONAR_HH

// ros stuff
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/advertise_options.h>

// ros messages stuff
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/fill_image.h>
#include <std_msgs/Float64.h>
#include <image_transport/image_transport.h>

// gazebo stuff
#include <sdf/Param.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>
#include <gazebo/msgs/MessageTypes.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/sensors/SensorTypes.hh>
#include <gazebo/plugins/DepthCameraPlugin.hh>

// dynamic reconfigure stuff
#include <gazebo_plugins/GazeboRosCameraConfig.h>
#include <dynamic_reconfigure/server.h>

// boost stuff
#include <boost/thread/mutex.hpp>

// camera stuff
#include <gazebo_plugins/gazebo_ros_camera_utils.h>

#include <opencv2/core.hpp>
#include <complex>
#include <valarray>
#include <sstream>
#include <chrono>
#include "fft.hpp"

namespace gazebo
{
  typedef std::complex<double> Complex;
  typedef std::valarray<Complex> CArray;
  typedef std::valarray<CArray> CArray2D;

  class NpsGazeboRosImageSonar : public SensorPlugin, GazeboRosCameraUtils
  {
    /// \brief Constructor
    /// \param parent The parent entity, must be a Model or a Sensor
    public: NpsGazeboRosImageSonar();

    /// \brief Destructor
    public: ~NpsGazeboRosImageSonar();

    /// \brief Load the plugin
    /// \param take in SDF root element
    public: virtual void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

    /// \brief Advertise point cloud and depth image
    public: virtual void Advertise();

    /// \brief Update the controller
    protected: virtual void OnNewDepthFrame(const float *_image,
                                            unsigned int _width,
                                            unsigned int _height,
                                            unsigned int _depth,
                                            const std::string &_format);

    /// \brief Update the controller
    protected: virtual void OnNewImageFrame(const unsigned char *_image,
                                            unsigned int _width,
                                            unsigned int _height,
                                            unsigned int _depth,
                                            const std::string &_format);

    /// \brief Compute a normal texture and implement sonar model
    private: void ComputeSonarImage(const float *_src);
    private: double ComputeIncidence(double azimuth, double elevation, cv::Vec3f normal);
    private: cv::Mat ComputeNormalImage(cv::Mat& depth);

    /// \brief Parameters for sonar properties
    private: double sonarFreq;
    private: double bandwidth;
    private: double freqResolution;
    private: double soundSpeed;
    private: bool constMu;
    private: double absorption;
    private: double attenuation;
    private: double mu; // surface reflectivity
    private: double fmin;
    private: double fmax;
    private: double df;
    private: int sonarCalcWidthSkips;
    private: int sonarCalcHeightSkips;
    private: int nBeams;
    private: int ray_nAzimuthRays;
    private: int ray_nElevationRays;

    /// \brief CSV log writing stream for verifications
    protected: std::ofstream writeLog;
    protected: u_int64_t writeCounter;
    protected: u_int64_t writeNumber;
    protected: u_int64_t writeInterval;
    protected: bool writeLogFlag;

    /// \brief Keep track of number of connctions for plugin outputs
    private: int depth_image_connect_count_;
    private: int depth_info_connect_count_;
    private: void DepthImageConnect();
    private: void DepthImageDisconnect();
    private: void DepthInfoConnect();
    private: void DepthInfoDisconnect();
    private: common::Time last_depth_image_camera_info_update_time_;

    /// \brief A pointer to the ROS node.  A node will be instantiated if it does not exist.
    private: ros::Publisher depth_image_pub_;

    private: sensor_msgs::Image depth_image_msg_;

    std::default_random_engine generator;

    private: void InfoConnect();
    private: void InfoDisconnect();

    using GazeboRosCameraUtils::PublishCameraInfo;
    protected: virtual void PublishCameraInfo();

    /// \brief image where each pixel contains the depth information
    private: std::string depth_image_topic_name_;
    private: std::string depth_image_camera_info_topic_name_;

    // overload with our own
    private: common::Time depth_sensor_update_time_;
    protected: ros::Publisher depth_image_camera_info_pub_;

    private: event::ConnectionPtr load_connection_;

    // from DepthCameraPlugin
    protected: unsigned int width, height, depth;
    protected: std::string format;

    protected: sensors::DepthCameraSensorPtr parentSensor;
    protected: rendering::DepthCameraPtr depthCamera;

    private: event::ConnectionPtr newDepthFrameConnection;
    private: event::ConnectionPtr newImageFrameConnection;
  };

}
#endif
