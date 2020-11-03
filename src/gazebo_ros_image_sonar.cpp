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

#include <algorithm>
#include <assert.h>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>

#include <nps_uw_sensors_gazebo/gazebo_ros_image_sonar.hh>
#include <gazebo/sensors/Sensor.hh>
#include <sdf/sdf.hh>
#include <gazebo/sensors/SensorTypes.hh>

#include <tf/tf.h>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include <nps_uw_sensors_gazebo/sonar_calculation_cuda.cuh>

namespace gazebo
{
// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(NpsGazeboRosImageSonar)


// Constructor
NpsGazeboRosImageSonar::NpsGazeboRosImageSonar() :
  SensorPlugin(), width(0), height(0), depth(0)
{
  this->depth_image_connect_count_ = 0;
  this->last_depth_image_camera_info_update_time_ = common::Time(0);

  // for csv write logs
  this->writeCounter = 0;
  this->writeNumber = 1;
}


// Destructor
NpsGazeboRosImageSonar::~NpsGazeboRosImageSonar()
{
  this->newDepthFrameConnection.reset();
  this->newImageFrameConnection.reset();

  this->parentSensor.reset();
  this->depthCamera.reset();
  
  // CSV log write stream close
  writeLog.close();
}


// Load the controller
void NpsGazeboRosImageSonar::Load(sensors::SensorPtr _parent, 
                                  sdf::ElementPtr _sdf)
{
  this->parentSensor =
    std::dynamic_pointer_cast<sensors::DepthCameraSensor>(_parent);
  this->depthCamera = this->parentSensor->DepthCamera();

  if (!this->parentSensor)
  {
    gzerr << "DepthCameraPlugin not attached to a depthCamera sensor\n";
    return;
  }

  this->width = this->depthCamera->ImageWidth();
  this->height = this->depthCamera->ImageHeight();
  this->depth = this->depthCamera->ImageDepth();
  this->format = this->depthCamera->ImageFormat();

  this->newDepthFrameConnection = 
    this->depthCamera->ConnectNewDepthFrame(
        std::bind(&NpsGazeboRosImageSonar::OnNewDepthFrame,
                  this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4,
                  std::placeholders::_5));

  this->newImageFrameConnection = 
    this->depthCamera->ConnectNewImageFrame(
        std::bind(&NpsGazeboRosImageSonar::OnNewImageFrame,
                  this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4,
                  std::placeholders::_5));

  this->parentSensor->SetActive(true);

  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM_NAMED("depth_camera", "A ROS node for Gazebo has not been initialized, unable to load plugin. "
        << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  // copying from DepthCameraPlugin into GazeboRosCameraUtils
  this->parentSensor_ = this->parentSensor;
  this->width_ = this->width;
  this->height_ = this->height;
  this->depth_ = this->depth;
  this->format_ = this->format;
  this->camera_ = this->depthCamera;

  // not using default GazeboRosCameraUtils topics
  if (!_sdf->HasElement("imageTopicName"))
    this->image_topic_name_ = "ir/image_raw";
  if (!_sdf->HasElement("cameraInfoTopicName"))
    this->camera_info_topic_name_ = "ir/camera_info";

  // depth image stuff
  if (!_sdf->HasElement("depthImageTopicName"))
    this->depth_image_topic_name_ = "depth/image_raw";
  else
    this->depth_image_topic_name_ = 
      _sdf->GetElement("depthImageTopicName")->Get<std::string>();

  if (!_sdf->HasElement("depthImageCameraInfoTopicName"))
    this->depth_image_camera_info_topic_name_ = "depth/camera_info";
  else
    this->depth_image_camera_info_topic_name_ =
      _sdf->GetElement("depthImageCameraInfoTopicName")->Get<std::string>();

  if (!_sdf->HasElement("clip"))
  {
    gzerr << "We do not have clip" << std::endl;
  }
  else
  {
    gzerr << "We do have clip" << std::endl;
    gzerr << _sdf->GetElement("clip")->GetElement("far")->Get<double>()
          << std::endl;
  }

  // TODO: Implement additional SDF options (ROS namespaces & topics) here

  // Read sonar properties from model.sdf
  if (!_sdf->HasElement("sonarFreq"))
    this->sonarFreq = 900e3;  // Blueview P900 [Hz]
  else
    this->sonarFreq =
      _sdf->GetElement("sonarFreq")->Get<double>();
  if (!_sdf->HasElement("bandwidth"))
    this->bandwidth = 29.5e6;  // Blueview P900 [Hz]
  else
    this->bandwidth =
      _sdf->GetElement("bandwidth")->Get<double>();
  if (!_sdf->HasElement("freqResolution"))
    this->freqResolution = 100e2;
  else
    this->freqResolution =
      _sdf->GetElement("freqResolution")->Get<double>();
  if (!_sdf->HasElement("soundSpeed"))
    this->soundSpeed = 1500;
  else
    this->soundSpeed =
      _sdf->GetElement("soundSpeed")->Get<double>();
  if (!_sdf->HasElement("constantReflectivity"))
    this->constMu = true;
  else
    this->constMu =
      _sdf->GetElement("constantReflectivity")->Get<bool>();
  if (!_sdf->HasElement("sonarCalcWidthSkips"))
    this->sonarCalcWidthSkips = 1;
  else
    this->sonarCalcWidthSkips =
      _sdf->GetElement("sonarCalcWidthSkips")->Get<int>();
  if (!_sdf->HasElement("sonarCalcHeightSkips"))
    this->sonarCalcHeightSkips = 1;
  else
    this->sonarCalcHeightSkips =
      _sdf->GetElement("sonarCalcHeightSkips")->Get<int>();

  // Calculate common sonar parameters
  this->fmin = this->sonarFreq - this->bandwidth/2.0*4.0;
  this->fmax = this->sonarFreq + this->bandwidth/2.0*4.0;
  // if (this->constMu)
  this->mu = 10e-4;
  // else
  //   // nothing yet

  // Transmission path properties (typical model used here)
  this->absorption = 0.0354; // [dB/m]
  this->attenuation = this->absorption*log(10)/20.0;
  
  // FOV, Number of beams, number of rays are defined at model.sdf
  // Currently, this->width equals # of beams, and this->height equals # of rays
  // Each beam consists of (elevation,azimuth)=(this->height,1) rays
  // Beam patterns
  this->nBeams = this->width;
  this->ray_nElevationRays = this->height;
  this->ray_nAzimuthRays = 1;

  // Print sonar calculation settings
  ROS_INFO_STREAM("");
  ROS_INFO_STREAM("==================================================");
  ROS_INFO_STREAM("============   SONAR PLUGIN LOADED   =============");
  ROS_INFO_STREAM("==================================================");
  ROS_INFO_STREAM("# of Beams = " << this->nBeams);
  ROS_INFO_STREAM("# of Rays/Beam (Elevation, Azimuth) = ("
      << ray_nElevationRays << ", " << ray_nAzimuthRays << ")");
  ROS_INFO_STREAM("Calculation skips (Elevation, Azimuth) = (" 
      << this->sonarCalcHeightSkips << ", " << this->sonarCalcWidthSkips << ")");
  ROS_INFO_STREAM("==================================================");
  ROS_INFO_STREAM("");

  // get writeLog Flag
  if (_sdf->HasElement("writeLog"))
  {
    this->writeLogFlag = _sdf->Get<bool>("writeLog");
    if (_sdf->HasElement("writeLog"))
      this->writeInterval = _sdf->Get<int>("writeFrameInterval");
    else
      this->writeInterval = 10;
    ROS_INFO_STREAM("Raw data at " << "/tmp/SonarRawData_{numbers}.csv");
    ROS_INFO_STREAM("every " << this->writeInterval << " frames"
                    << "for " << floor(nBeams/sonarCalcWidthSkips) << " beams");
    system("rm /tmp/SonarRawData*.csv");
  }

  load_connection_ = 
    GazeboRosCameraUtils::OnLoad(boost::bind(&NpsGazeboRosImageSonar::Advertise, this));
  GazeboRosCameraUtils::Load(_parent, _sdf);
}


void NpsGazeboRosImageSonar::Advertise()
{
  ros::AdvertiseOptions depth_image_ao =
    ros::AdvertiseOptions::create<sensor_msgs::Image>(
      this->depth_image_topic_name_, 1,
      boost::bind(&NpsGazeboRosImageSonar::DepthImageConnect, this),
      boost::bind(&NpsGazeboRosImageSonar::DepthImageDisconnect, this),
      ros::VoidPtr(), &this->camera_queue_);
  this->depth_image_pub_ = this->rosnode_->advertise(depth_image_ao);

  ros::AdvertiseOptions depth_image_camera_info_ao =
    ros::AdvertiseOptions::create<sensor_msgs::CameraInfo>(
        this->depth_image_camera_info_topic_name_, 1,
        boost::bind(&NpsGazeboRosImageSonar::DepthInfoConnect, this),
        boost::bind(&NpsGazeboRosImageSonar::DepthInfoDisconnect, this),
        ros::VoidPtr(), &this->camera_queue_);
  this->depth_image_camera_info_pub_ =
    this->rosnode_->advertise(depth_image_camera_info_ao);
}


//----------------------------------------------------------------
// Increment and decriment a connection counter so that the sensor
// is only active and ROS messages being published when required
// TODO: Update once new message (plugin output) is being published
//----------------------------------------------------------------

void NpsGazeboRosImageSonar::DepthImageConnect()
{
  this->depth_image_connect_count_++;
  this->parentSensor->SetActive(true);
}

void NpsGazeboRosImageSonar::DepthImageDisconnect()
{
  this->depth_image_connect_count_--;
}

void NpsGazeboRosImageSonar::DepthInfoConnect()
{
  this->depth_info_connect_count_++;
}

void NpsGazeboRosImageSonar::DepthInfoDisconnect()
{
  this->depth_info_connect_count_--;
}


// Update everything when Gazebo provides a new depth frame (texture)
void NpsGazeboRosImageSonar::OnNewDepthFrame(const float *_image,
                                             unsigned int _width,
                                             unsigned int _height,
                                             unsigned int _depth,
                                             const std::string &_format)
{
  if (!this->initialized_ || this->height_ <=0 || this->width_ <=0)
    return;

  this->depth_sensor_update_time_ = this->parentSensor->LastMeasurementTime();

  if (this->parentSensor->IsActive())
  {
    // Deactivate if no subscribers
    if (this->depth_image_connect_count_ <= 0 &&
        (*this->image_connect_count_) <= 0)
    {
      this->parentSensor->SetActive(false);
    }
    else
    {
      // Generate depth image data if topics have subscribers
      if (this->depth_image_connect_count_ > 0)
        this->ComputeSonarImage(_image);
    }
  }
  else
  {
    // Won't this just toggle on and off unnecessarily?
    // TODO: Find a better way to make sure it runs once after activation
    if (this->depth_image_connect_count_ <= 0)
      // do this first so there's chance for sensor to run 1 frame after activate
      this->parentSensor->SetActive(true);
  }
}


// Process the camera image when Gazebo provides one. Do we actually need this?
void NpsGazeboRosImageSonar::OnNewImageFrame(const unsigned char *_image,
                                             unsigned int _width,
                                             unsigned int _height,
                                             unsigned int _depth,
                                             const std::string &_format)
{
  if (!this->initialized_ || this->height_ <=0 || this->width_ <=0)
    return;

  this->sensor_update_time_ = this->parentSensor->LastMeasurementTime();

  if (!this->parentSensor->IsActive())
  {
    if ((*this->image_connect_count_) > 0)
      // do this first so there's chance for sensor to run 1 frame after activate
      this->parentSensor->SetActive(true);
  }
  else
  {
    if ((*this->image_connect_count_) > 0)
    {
      this->PutCameraData(_image);
    }
  }
}

// Most of the plugin work happens here
void NpsGazeboRosImageSonar::ComputeSonarImage(const float *_src)
{
  this->lock_.lock();

  // Use OpenCV to compute a normal image from the depth image
  cv::Mat depth_image(this->height, this->width, CV_32FC1, (float*)_src);
  cv::Mat normal_image = this->ComputeNormalImage(depth_image);
  double vFOV = this->parentSensor->DepthCamera()->VFOV().Radian();
  double hFOV = this->parentSensor->DepthCamera()->HFOV().Radian();
  double vPixelSize = vFOV / this->height;
  double hPixelSize = hFOV / this->width;

  // For calc time measure
  auto start = std::chrono::high_resolution_clock::now();
  // ------------------------------------------------//
  // --------      Sonar calculations       -------- //
  // ------------------------------------------------//
  CArray2D P_Beams = NpsGazeboSonar::sonar_calculation_wrapper(
                  depth_image,   // cv::Mat& depth_image
									normal_image,  // cv::Mat& normal_image
                  hPixelSize,    // hPixelSize
                  vPixelSize,    // vPixelSize
                  hFOV,          // hFOV
                  vFOV,          // VFOV
									hPixelSize,    // _beam_elevationAngleWidth
									vPixelSize,    // _beam_azimuthAngleWidth
									hPixelSize/(this->ray_nElevationRays-1),  // _ray_elevationAngleWidth
									vPixelSize,    // _ray_azimuthAngleWidth
									this->soundSpeed,    // _soundSpeed
									this->nBeams,        // _nBeams
									this->sonarFreq,     // _sonarFreq
								  this->fmax,          // _fmax
                  this->fmin,          // _fmin
                  this->bandwidth,     // _bandwidth
                  this->mu,            // _mu
                  this->attenuation);  // _attenuation

  // For calc time measure
  auto stop = std::chrono::high_resolution_clock::now(); 
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  ROS_INFO_STREAM(duration.count()/10000 << "/ 100 [s]");

  // CSV log write stream
  // Each columns corresponds to each beams
  if (this->writeLogFlag)
  { 
    this->writeCounter = this->writeCounter + 1;
    if (this->writeCounter % this->writeInterval == 0)
    {
      double time = this->parentSensor_->LastMeasurementTime().Double();
      std::stringstream filename;
      filename << "/tmp/SonarRawData_" << std::setw(6) <<  std::setfill('0') 
               << this->writeNumber << ".csv"; 
      writeLog.open(filename.str().c_str(), std::ios_base::app);
      filename.clear();
      writeLog << "# Raw Sonar Data Log (Row: beams, colmns: time series data)\n";
      writeLog << "#  nBeams : " << floor(nBeams/sonarCalcWidthSkips) << "\n";
      writeLog << "# Simulation time : " << time << "\n";
      for (size_t i = 0; i < P_Beams[0].size(); i++)
      {
        // writeLog << sqrt(pow(P_Beams[0][i].real(),2) + pow(P_Beams[0][i].imag(),2));
        writeLog << P_Beams[0][i].real() << " + "  << P_Beams[0][i].imag() << "i";
        for (size_t b = 0; b < nBeams; b += this->sonarCalcWidthSkips)
        { 
          if (b != 0)
            writeLog << "," << P_Beams[b][i].real() << " + "<< P_Beams[b][i].imag() << "i";
            // writeLog << "," << sqrt(pow(P_Beams[b][i].real(),2) + pow(P_Beams[b][i].imag(),2));
        }
        writeLog << "\n";
      }
      writeLog.close();

      this->writeNumber = this->writeNumber + 1;
    }
  }
  // ----- End of sonar calculation
  
  // Still publishing the depth image (just because)
  this->depth_image_msg_.header.frame_id = this->frame_name_;
  this->depth_image_msg_.header.stamp.sec = this->depth_sensor_update_time_.sec;
  this->depth_image_msg_.header.stamp.nsec = this->depth_sensor_update_time_.nsec;
  cv_bridge::CvImage img_bridge;
  img_bridge = cv_bridge::CvImage(this->depth_image_msg_.header, 
                                  sensor_msgs::image_encodings::TYPE_32FC1,
                                  depth_image);
  img_bridge.toImageMsg(this->depth_image_msg_); // from cv_bridge to sensor_msgs::Image
  this->depth_image_pub_.publish(this->depth_image_msg_);

  this->lock_.unlock();
}


/////////////////////////////////////////////////
// incidence angle is target's normal angle accounting for the ray's azimuth
// and elevation
double NpsGazeboRosImageSonar::ComputeIncidence(double azimuth, double elevation, cv::Vec3f normal)
{
  // ray normal from camera azimuth and elevation
  double camera_x = cos(-azimuth)*cos(elevation);
  double camera_y = sin(-azimuth)*cos(elevation);
  double camera_z = sin(elevation);
  cv::Vec3f ray_normal(camera_x, camera_y, camera_z);

  // target normal with axes compensated to camera axes
  cv::Vec3f target_normal(normal[2], -normal[0], -normal[1]);

  // dot product
  double dot_product = ray_normal.dot(target_normal);
  return M_PI - acos(dot_product);
}


/////////////////////////////////////////////////
cv::Mat NpsGazeboRosImageSonar::ComputeNormalImage(cv::Mat& depth)
{
  // filters
  cv::Mat_<float> f1 = (cv::Mat_<float>(3, 3) << 1,  2,  1,
                                                 0,  0,  0,
                                                -1, -2, -1) / 8;

  cv::Mat_<float> f2 = (cv::Mat_<float>(3, 3) << 1, 0, -1,
                                                 2, 0, -2,
                                                 1, 0, -1) / 8;

  cv::Mat f1m, f2m;
  cv::flip(f1, f1m, 0);
  cv::flip(f2, f2m, 1);

  cv::Mat n1, n2;
  cv::filter2D(depth, n1, -1, f1m, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(depth, n2, -1, f2m, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

  cv::Mat no_readings;
  cv::erode(depth == 0, no_readings, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
  //cv::dilate(no_readings, no_readings, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
  n1.setTo(0, no_readings);
  n2.setTo(0, no_readings);

  std::vector<cv::Mat> images(3);
  cv::Mat white = cv::Mat::ones(depth.rows, depth.cols, CV_32FC1);

  // NOTE: with different focal lengths, the expression becomes
  // (-dzx*fy, -dzy*fx, fx*fy)
  images.at(0) = n1;    //for green channel
  images.at(1) = n2;    //for red channel
  images.at(2) = 1.0/this->focal_length_*depth; //for blue channel

  cv::Mat normal_image;
  cv::merge(images, normal_image);
 
  for (int i = 0; i < normal_image.rows; ++i)
  {
    for (int j = 0; j < normal_image.cols; ++j)
    {
      cv::Vec3f& n = normal_image.at<cv::Vec3f>(i, j);
      n = cv::normalize(n);
      float& d = depth.at<float>(i, j);
    }
  }
  return normal_image;
}


/////////////////////////////////////////////////
void NpsGazeboRosImageSonar::PublishCameraInfo()
{
  ROS_DEBUG_NAMED("depth_camera", "publishing default camera info, then depth camera info");
  GazeboRosCameraUtils::PublishCameraInfo();

  if (this->depth_info_connect_count_ > 0)
  {
    common::Time sensor_update_time = this->parentSensor_->LastMeasurementTime();

    this->sensor_update_time_ = sensor_update_time;
    if (sensor_update_time - this->last_depth_image_camera_info_update_time_ >= this->update_period_)
    {
      this->PublishCameraInfo(this->depth_image_camera_info_pub_);  // , sensor_update_time);
      this->last_depth_image_camera_info_update_time_ = sensor_update_time;
    }
  }
}

}
