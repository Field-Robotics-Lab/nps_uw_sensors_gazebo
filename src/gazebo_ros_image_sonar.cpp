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
}


// Destructor
NpsGazeboRosImageSonar::~NpsGazeboRosImageSonar()
{
  this->newDepthFrameConnection.reset();
  this->newImageFrameConnection.reset();

  this->parentSensor.reset();
  this->depthCamera.reset();
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
  // The others are just for convenience
  cv::Mat depth_image(this->height, this->width, CV_32FC1, (float*)_src);
  cv::Mat normal_image = this->ComputeNormalImage(depth_image);
  double vFOV = this->parentSensor->DepthCamera()->VFOV().Radian();
  double hFOV = this->parentSensor->DepthCamera()->HFOV().Radian();
  double vPixelSize = vFOV / this->height;
  double hPixelSize = hFOV / this->width;

  // INSERT MODEL HERE
  // The loops are just an example of how the depth and normals
  // data can be  accessed and basic values calculated
  for (size_t row = 0; row < this->height; row++)
  {
      // Calculate the elevation as the vertical center of the pixel
      // Assumes the texture's origin is at the upper left corner
      double elevation = (vFOV/2.0) - row * vPixelSize - vPixelSize/2.0;
      for (size_t col = 0; col < this->height; col++)
      {
          // Caluculate azimuth as the horizontal center of the pixel
          double azimuth = -(hFOV/2.0) + col * hPixelSize + hPixelSize/2.0;

          // Depth is a floating point at the texture pixel location
          float depth = depth_image.at<float>(row, col);

          // Normal is a 3-vector (float) at texture pixel location
          cv::Vec3f& normal = normal_image.at<cv::Vec3f>(row, col);
      }
  }

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
