/*
 * Copyright 2020 Naval Postgraduate School
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
*/

#include <assert.h>
#include <sys/stat.h>
#include <tf/tf.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/point_cloud2_iterator.h>

#include <nps_uw_sensors_gazebo/sonar_calculation_cuda.cuh>

#include <opencv2/core/core.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>

#include <nps_uw_sensors_gazebo/gazebo_ros_image_sonar.hh>
#include <gazebo/sensors/Sensor.hh>
#include <sdf/sdf.hh>
#include <gazebo/sensors/SensorTypes.hh>

#include <algorithm>
#include <string>
#include <vector>
#include <limits>

namespace gazebo
{
// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(NpsGazeboRosImageSonar)


// Constructor
NpsGazeboRosImageSonar::NpsGazeboRosImageSonar() :
  SensorPlugin(), width(0), height(0), depth(0)
{
  this->depth_image_connect_count_ = 0;
  this->depth_info_connect_count_ = 0;
  this->point_cloud_connect_count_ = 0;
  this->sonar_image_connect_count_ = 0;
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
  this->newRGBPointCloudConnection.reset();
  this->newSonarImageConnection.reset();

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

  this->newSonarImageConnection =
    this->depthCamera->ConnectNewDepthFrame(
        std::bind(&NpsGazeboRosImageSonar::OnNewDepthFrame,
                  this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4,
                  std::placeholders::_5));

  this->parentSensor->SetActive(true);

  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM_NAMED("depth_camera", "A ROS node for Gazebo "
        << "has not been initialized, unable to load plugin. "
        << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so'"
        << " in the gazebo_ros package)");
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

  if (!_sdf->HasElement("pointCloudTopicName"))
    this->point_cloud_topic_name_ = "points";
  else
    this->point_cloud_topic_name_ =
        _sdf->GetElement("pointCloudTopicName")->Get<std::string>();

  if (!_sdf->HasElement("pointCloudCutoff"))
    this->point_cloud_cutoff_ = 0.4;
  else
    this->point_cloud_cutoff_ =
        _sdf->GetElement("pointCloudCutoff")->Get<double>();

  // sonar stuff
  if (!_sdf->HasElement("sonarImageRawTopicName"))
    this->sonar_image_raw_topic_name_ = "sonar_image_raw";
  else
    this->sonar_image_raw_topic_name_ =
      _sdf->GetElement("sonarImageRawTopicName")->Get<std::string>();
  if (!_sdf->HasElement("sonarImageTopicName"))
    this->sonar_image_topic_name_ = "sonar_image";
  else
    this->sonar_image_topic_name_ =
      _sdf->GetElement("sonarImageTopicName")->Get<std::string>();


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
  if (!_sdf->HasElement("soundSpeed"))
    this->soundSpeed = 1500;
  else
    this->soundSpeed =
      _sdf->GetElement("soundSpeed")->Get<double>();
  if (!_sdf->HasElement("maxDistance"))
    this->maxDistance = 60;
  else
    this->maxDistance =
      _sdf->GetElement("maxDistance")->Get<double>();
  if (!_sdf->HasElement("sourceLevel"))
    this->sourceLevel = 220;
  else
    this->sourceLevel =
      _sdf->GetElement("sourceLevel")->Get<double>();
  if (!_sdf->HasElement("constantReflectivity"))
    this->constMu = true;
  else
    this->constMu =
      _sdf->GetElement("constantReflectivity")->Get<bool>();
  if (!_sdf->HasElement("raySkips"))
    this->raySkips = 10;
  else
    this->raySkips =
      _sdf->GetElement("raySkips")->Get<int>();
  // Configure skips
  if (this->raySkips == 0) this->raySkips = 1;

  // --- Calculate common sonar parameters ---- //
  // if (this->constMu)
  this->mu = 1e-3;
  // else
  //   // nothing yet

  // Transmission path properties (typical model used here)
  // More sophisticated model by Francois-Garrison model is available
  this->absorption = 0.0354;  // [dB/m]
  this->attenuation = this->absorption*log(10)/20.0;

  // Range vector
  const float max_T = this->maxDistance*2.0/this->soundSpeed;
  float delta_f = 1.0/max_T;
  const float delta_t = 1.0/this->bandwidth;
  this->nFreq = ceil(this->bandwidth/delta_f);
  delta_f = this->bandwidth/this->nFreq;
  const int nTime = nFreq;
  this->rangeVector = new float[nTime];
  for (int i = 0; i < nTime; i++)
  {
    this->rangeVector[i] = delta_t*i*this->soundSpeed/2.0;
  }

  // FOV, Number of beams, number of rays are defined at model.sdf
  // Currently, this->width equals # of beams, and this->height equals # of rays
  // Each beam consists of (elevation,azimuth)=(this->height,1) rays
  // Beam patterns
  this->nBeams = this->width;
  this->nRays = this->height;
  this->ray_nElevationRays = this->height;
  this->ray_nAzimuthRays = 1;

  // Print sonar calculation settings
  ROS_INFO_STREAM("");
  ROS_INFO_STREAM("==================================================");
  ROS_INFO_STREAM("============   SONAR PLUGIN LOADED   =============");
  ROS_INFO_STREAM("==================================================");
  ROS_INFO_STREAM("Maximum view range  [m] = " << this->maxDistance);
  ROS_INFO_STREAM("Distance resolution [m] = " <<
                    this->soundSpeed*(1.0/(this->nFreq*delta_f)));
  ROS_INFO_STREAM("# of Beams = " << this->nBeams);
  ROS_INFO_STREAM("# of Rays / Beam (Elevation, Azimuth) = ("
      << ray_nElevationRays << ", " << ray_nAzimuthRays << ")");
  ROS_INFO_STREAM("Calculation skips (Elevation) = "
      << this->raySkips);
  ROS_INFO_STREAM("# of Time data / Beam = " << this->nFreq);
  ROS_INFO_STREAM("==================================================");
  ROS_INFO_STREAM("");

  // get writeLog Flag
  if (!_sdf->HasElement("writeLog"))
    this->writeLogFlag = false;
  else
  {
    this->writeLogFlag = _sdf->Get<bool>("writeLog");
    if (this->writeLogFlag)
    {
      if (_sdf->HasElement("writeFrameInterval"))
        this->writeInterval = _sdf->Get<int>("writeFrameInterval");
      else
        this->writeInterval = 10;
      ROS_INFO_STREAM("Raw data at " << "/tmp/SonarRawData_{numbers}.csv");
      ROS_INFO_STREAM("every " << this->writeInterval << " frames");
      ROS_INFO_STREAM("");

      struct stat buffer;
      std::string logfilename("/tmp/SonarRawData_000001.csv");
      if (stat (logfilename.c_str(), &buffer) == 0)
        system("rm /tmp/SonarRawData*.csv");
    }
  }

  // Get debug flag for computation time display
  if (!_sdf->HasElement("debugFlag"))
    this->debugFlag = false;
  else
    this->debugFlag =
      _sdf->GetElement("debugFlag")->Get<bool>();

  // -- Pre calculations for sonar -- //
  // Hamming window
  this->window = new float[this->nFreq];
  float windowSum = 0;
  for (size_t f = 0; f < this->nFreq; f++)
  {
    this->window[f] = 0.54 - 0.46 * cos(2.0*M_PI*(f+1)/this->nFreq);
    windowSum += pow(this->window[f], 2.0);
  }
  for (size_t f = 0; f < this->nFreq; f++)
    this->window[f] = this->window[f]/sqrt(windowSum);

  // Sonar corrector preallocation
  this->beamCorrector = new float*[nBeams];
  for (int i = 0; i < nBeams; i++)
      this->beamCorrector[i] = new float[nBeams];
  this->beamCorrectorSum = 0.0;

  load_connection_ =
    GazeboRosCameraUtils::OnLoad(
            boost::bind(&NpsGazeboRosImageSonar::Advertise, this));
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

  ros::AdvertiseOptions normal_image_ao =
    ros::AdvertiseOptions::create<sensor_msgs::Image>(
      this->depth_image_topic_name_+"_normals", 1,
      boost::bind(&NpsGazeboRosImageSonar::NormalImageConnect, this),
      boost::bind(&NpsGazeboRosImageSonar::NormalImageDisconnect, this),
      ros::VoidPtr(), &this->camera_queue_);
  this->normal_image_pub_ = this->rosnode_->advertise(normal_image_ao);

  ros::AdvertiseOptions point_cloud_ao =
    ros::AdvertiseOptions::create<sensor_msgs::PointCloud2>(
      this->point_cloud_topic_name_, 1,
      boost::bind(&NpsGazeboRosImageSonar::PointCloudConnect, this),
      boost::bind(&NpsGazeboRosImageSonar::PointCloudDisconnect, this),
      ros::VoidPtr(), &this->camera_queue_);
  this->point_cloud_pub_ = this->rosnode_->advertise(point_cloud_ao);

  // Publisher for sonar image
  ros::AdvertiseOptions sonar_image_raw_ao =
    ros::AdvertiseOptions::create<imaging_sonar_msgs::SonarImage>(
      this->sonar_image_raw_topic_name_, 1,
      boost::bind(&NpsGazeboRosImageSonar::SonarImageRawConnect, this),
      boost::bind(&NpsGazeboRosImageSonar::SonarImageRawDisconnect, this),
      ros::VoidPtr(), &this->camera_queue_);
  this->sonar_image_raw_pub_ = this->rosnode_->advertise(sonar_image_raw_ao);

  ros::AdvertiseOptions sonar_image_ao =
    ros::AdvertiseOptions::create<sensor_msgs::Image>(
      this->sonar_image_topic_name_, 1,
      boost::bind(&NpsGazeboRosImageSonar::SonarImageConnect, this),
      boost::bind(&NpsGazeboRosImageSonar::SonarImageDisconnect, this),
      ros::VoidPtr(), &this->camera_queue_);
  this->sonar_image_pub_ = this->rosnode_->advertise(sonar_image_ao);


  this->sonar_image_pub_ =
      this->rosnode_->advertise<sensor_msgs::Image>
      ("sonar_image", 10);
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

void NpsGazeboRosImageSonar::NormalImageConnect()
{
  this->depth_image_connect_count_++;
  this->parentSensor->SetActive(true);
}

void NpsGazeboRosImageSonar::NormalImageDisconnect()
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

void NpsGazeboRosImageSonar::PointCloudConnect()
{
  this->point_cloud_connect_count_++;
  (*this->image_connect_count_)++;
  this->parentSensor->SetActive(true);
}

void NpsGazeboRosImageSonar::PointCloudDisconnect()
{
  this->point_cloud_connect_count_--;
  (*this->image_connect_count_)--;
  if (this->point_cloud_connect_count_ <= 0)
    this->parentSensor->SetActive(false);
}
void NpsGazeboRosImageSonar::SonarImageConnect()
{
  this->sonar_image_connect_count_++;
  this->parentSensor->SetActive(true);
}
void NpsGazeboRosImageSonar::SonarImageDisconnect()
{
  this->sonar_image_connect_count_--;
}
void NpsGazeboRosImageSonar::SonarImageRawConnect()
{
  this->sonar_image_connect_count_++;
  this->parentSensor->SetActive(true);
}
void NpsGazeboRosImageSonar::SonarImageRawDisconnect()
{
  this->sonar_image_connect_count_--;
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
        this->point_cloud_connect_count_ <= 0 &&
        (*this->image_connect_count_) <= 0)
    {
      this->parentSensor->SetActive(false);
    }
    else
    {
      // Generate a point cloud every time regardless of subscriptions
      // for use in sonar computation (published in function if needed)
      this->ComputePointCloud(_image);

      // Generate sonar image data if topics have subscribers
      if (this->depth_image_connect_count_ > 0)
        this->ComputeSonarImage(_image);
    }
  }
  else
  {
    // Won't this just toggle on and off unnecessarily?
    // TODO: Find a better way to ensure it runs once after activation
    if (this->depth_image_connect_count_ <= 0 ||
        this->point_cloud_connect_count_ > 0)
      // do this first so there's chance for sensor
      // to run 1 frame after activate
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
      // do this first so there's chance for sensor
      // to run 1 frame after activate
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
  cv::Mat depth_image = this->point_cloud_image_;
  cv::Mat normal_image = this->ComputeNormalImage(depth_image);
  double vFOV = this->parentSensor->DepthCamera()->VFOV().Radian();
  double hFOV = this->parentSensor->DepthCamera()->HFOV().Radian();
  double vPixelSize = vFOV / this->height;
  double hPixelSize = hFOV / this->width;

  // rand number generator
  cv::Mat rand_image = cv::Mat(depth_image.rows, depth_image.cols, CV_32FC2);
  uint64 randN = static_cast<uint64>(std::rand());
  cv::theRNG().state = randN;
  cv::RNG rng = cv::theRNG();
  rng.fill(rand_image, cv::RNG::NORMAL, 0.f, 1.f);

  if (this->beamCorrectorSum == 0)
    ComputeCorrector();

  // For calc time measure
  auto start = std::chrono::high_resolution_clock::now();
  // ------------------------------------------------//
  // --------      Sonar calculations       -------- //
  // ------------------------------------------------//
  CArray2D P_Beams = NpsGazeboSonar::sonar_calculation_wrapper(
                  depth_image,   // cv::Mat& depth_image
                  normal_image,  // cv::Mat& normal_image
                  rand_image,    // cv::Mat& rand_image
                  hPixelSize,    // hPixelSize
                  vPixelSize,    // vPixelSize
                  hFOV,          // hFOV
                  vFOV,          // VFOV
                  hPixelSize,    // _beam_azimuthAngleWidth
                  vPixelSize,    // _beam_elevationAngleWidth
                  hPixelSize,    // _ray_azimuthAngleWidth
                  vPixelSize*(raySkips),  // _ray_elevationAngleWidth
                  this->soundSpeed,    // _soundSpeed
                  this->maxDistance,   // _maxDistance
                  this->sourceLevel,   // _sourceLevel
                  this->nBeams,        // _nBeams
                  this->nRays,         // _nRays
                  this->raySkips,      // _raySkips
                  this->sonarFreq,     // _sonarFreq
                  this->bandwidth,     // _bandwidth
                  this->nFreq,         // _nFreq
                  this->mu,            // _mu
                  this->attenuation,   // _attenuation
                  this->window,        // _window
                  this->beamCorrector,      // _beamCorrector
                  this->beamCorrectorSum,   // _beamCorrectorSum
                  this->debugFlag);

  // For calc time measure
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<
                  std::chrono::microseconds>(stop - start);
  if (debugFlag)
  {
    ROS_INFO_STREAM("GPU Sonar Frame Calc Time " <<
                    duration.count()/10000 << "/100 [s]\n");
  }

  // CSV log write stream
  // Each cols corresponds to each beams
  if (this->writeLogFlag)
  {
    this->writeCounter = this->writeCounter + 1;
    if (this->writeCounter == 1
        ||this->writeCounter % this->writeInterval == 0)
    {
      double time = this->parentSensor_->LastMeasurementTime().Double();
      std::stringstream filename;
      filename << "/tmp/SonarRawData_" << std::setw(6) <<  std::setfill('0')
               << this->writeNumber << ".csv";
      writeLog.open(filename.str().c_str(), std::ios_base::app);
      filename.clear();
      writeLog << "# Raw Sonar Data Log (Row: beams, Col: time series data)\n";
      writeLog << "# First column is range vector\n";
      writeLog << "#  nBeams : " << nBeams << "\n";
      writeLog << "# Simulation time : " << time << "\n";
      for (size_t i = 0; i < P_Beams[0].size(); i++)
      {
        // writing range vector at first column
        writeLog << this->rangeVector[i];
        for (size_t b = 0; b < nBeams; b ++)
        {
          if (P_Beams[b][i].imag() > 0)
            writeLog << "," << P_Beams[b][i].real()
                     << "+" << P_Beams[b][i].imag() << "i";
          else
            writeLog << "," << P_Beams[b][i].real()
                     << P_Beams[b][i].imag() << "i";
        }
        writeLog << "\n";
      }
      writeLog.close();

      this->writeNumber = this->writeNumber + 1;
    }
  }

  // Sonar image ROS msg
  this->sonar_image_raw_msg_.header.frame_id
        = this->frame_name_.c_str();
  this->sonar_image_raw_msg_.header.stamp.sec
        = this->depth_sensor_update_time_.sec;
  this->sonar_image_raw_msg_.header.stamp.nsec
        = this->depth_sensor_update_time_.nsec;
  this->sonar_image_raw_msg_.frequency = this->sonarFreq;
  this->sonar_image_raw_msg_.sound_speed = this->soundSpeed;
  this->sonar_image_raw_msg_.azimuth_beamwidth = hPixelSize;
  this->sonar_image_raw_msg_.elevation_beamwidth = hPixelSize*this->nRays;
  std::vector<float> azimuth_angles;
  double fl = static_cast<double>(width) / (2.0 * tan(hFOV/2.0));
  for (size_t beam = 0; beam < nBeams; beam ++)
    azimuth_angles.push_back(atan2(static_cast<double>(beam) -
                    0.5 * static_cast<double>(width-1), fl));
  this->sonar_image_raw_msg_.azimuth_angles = azimuth_angles;
  // std::vector<float> elevation_angles;
  // elevation_angles.push_back(vFOV / 2.0);  // 1D in elevation
  // this->sonar_image_raw_msg_.elevation_angles = elevation_angles;
  std::vector<float> ranges;
  for (size_t i = 0; i < P_Beams[0].size(); i ++)
    ranges.push_back(rangeVector[i]);
  this->sonar_image_raw_msg_.ranges = ranges;

  // this->sonar_image_raw_msg_.is_bigendian = false;
  this->sonar_image_raw_msg_.data_size = 1; //sizeof(float) * nFreq * nBeams;
  std::vector<uchar> intensities;
  for (size_t f = 0; f < nFreq; f ++)
    for (size_t beam = 0; beam < nBeams; beam ++)
      intensities.push_back(static_cast<uchar>(static_cast<int>(abs(P_Beams[beam][f]))));
  this->sonar_image_raw_msg_.intensities = intensities;

  this->sonar_image_raw_pub_.publish(this->sonar_image_raw_msg_);


  // Construct visual sonar image for rqt plot in sensor::image msg format
  cv_bridge::CvImage img_bridge;
  // Calculate and allocate plot data
  float Intensity[nBeams*4][nFreq];
  for (size_t beam = 0; beam < nBeams*4; beam ++)
  {
    for (size_t f = 0; f < nFreq; f ++)
    {
      if (beam > 2.5*nBeams &&beam < 3.5*nBeams)
        Intensity[beam][f] = (float)(abs(P_Beams[beam - 2.5*nBeams][f]));
      else
        Intensity[beam][f] = 0.0f;
    }
  }

  // Generate image
  cv::Mat Intensity_image(nBeams*4, nFreq, CV_32FC1, Intensity);

  // Rescale
  double minVal, maxVal;
  cv::minMaxLoc(Intensity_image, &minVal, &maxVal);
  Intensity_image -= minVal;
  Intensity_image *= 1./(maxVal - minVal);

  // Polar coordinate transform
  cv::Mat Intensity_image_polar;
  cv::Point2f center( (float)Intensity_image.cols/2.0, (float)Intensity_image.rows/2.0);
  double maxRadius = cv::min(center.y, center.x);
  cv::linearPolar(Intensity_image, Intensity_image_polar, center, maxRadius,
                cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS + cv::WARP_INVERSE_MAP);

  // Search for crop range
  int top = 0, bottom = 0, left = 0, right = 0;
  for (size_t i = 0; i < Intensity_image_polar.rows; i ++)
  {
    for (size_t j = 0; j < Intensity_image_polar.cols; j ++)
    {
      float data = Intensity_image_polar.at<float>(i, j);
      if (data != 0) {top = i; break;}
    }
    if (top !=0) {break;}
  }
  for (size_t i = Intensity_image_polar.rows-1; i > 0; i --)
  {
    for (size_t j = Intensity_image_polar.cols-1; j > 0; j --)
    {
      float data = Intensity_image_polar.at<float>(i, j);
      if (data != 0) {bottom = i; break;}
    }
    if (bottom !=0) {break;}
  }
  for (size_t j = 0; j < Intensity_image_polar.cols; j ++)
  {
    for (size_t i = 0; i < Intensity_image_polar.rows; i ++)
    {
      float data = Intensity_image_polar.at<float>(i, j);
      if (data != 0) {left = j; break;}
    }
    if (left !=0) {break;}
  }
  for (size_t j = Intensity_image_polar.cols-1; j > 0; j --)
  {
    for (size_t i = Intensity_image_polar.rows-1; i > 0; i --)
    {
      float data = Intensity_image_polar.at<float>(i, j);
      if (data != 0) {right = j; break;}
    }
    if (right !=0) {break;}
  }

  // Crop image
  cv::Mat Intensity_image_polar_cropped = Intensity_image_polar(
          cv::Rect(left, top, (float)(right-left), (float)(bottom-top)));

  // Publish final sonar image
  this->sonar_image_msg_.header.frame_id
        = this->frame_name_;
  this->sonar_image_msg_.header.stamp.sec
        = this->depth_sensor_update_time_.sec;
  this->sonar_image_msg_.header.stamp.nsec
        = this->depth_sensor_update_time_.nsec;
  img_bridge = cv_bridge::CvImage(this->sonar_image_msg_.header,
                                  sensor_msgs::image_encodings::TYPE_32FC1,
                                  Intensity_image_polar_cropped);
  img_bridge.toImageMsg(this->sonar_image_msg_); // from cv_bridge to sensor_msgs::Image

  this->sonar_image_pub_.publish(this->sonar_image_msg_);

  // ---------------------------------------- End of sonar calculation


  // Still publishing the depth and normal image (just because)
  // Depth image
  this->depth_image_msg_.header.frame_id
        = this->frame_name_;
  this->depth_image_msg_.header.stamp.sec
        = this->depth_sensor_update_time_.sec;
  this->depth_image_msg_.header.stamp.nsec
        = this->depth_sensor_update_time_.nsec;
  img_bridge = cv_bridge::CvImage(this->depth_image_msg_.header,
                                  sensor_msgs::image_encodings::TYPE_32FC1,
                                  depth_image);
  // from cv_bridge to sensor_msgs::Image
  img_bridge.toImageMsg(this->depth_image_msg_);
  this->depth_image_pub_.publish(this->depth_image_msg_);

  // Normal image
  this->normal_image_msg_.header.frame_id
        = this->frame_name_;
  this->normal_image_msg_.header.stamp.sec
        = this->depth_sensor_update_time_.sec;
  this->normal_image_msg_.header.stamp.nsec
        = this->depth_sensor_update_time_.nsec;
  cv::Mat normal_image8;
  normal_image.convertTo(normal_image8, CV_8UC3, 255.0);
  img_bridge = cv_bridge::CvImage(this->normal_image_msg_.header,
                                  sensor_msgs::image_encodings::RGB8,
                                  normal_image8);
  img_bridge.toImageMsg(this->normal_image_msg_);
  // from cv_bridge to sensor_msgs::Image
  this->normal_image_pub_.publish(this->normal_image_msg_);

  this->lock_.unlock();
}


void NpsGazeboRosImageSonar::ComputePointCloud(const float *_src)
{
  this->lock_.lock();

  this->point_cloud_msg_.header.frame_id
        = this->frame_name_;
  this->point_cloud_msg_.header.stamp.sec
        = this->depth_sensor_update_time_.sec;
  this->point_cloud_msg_.header.stamp.nsec
        = this->depth_sensor_update_time_.nsec;
  this->point_cloud_msg_.width = this->width;
  this->point_cloud_msg_.height = this->height;
  this->point_cloud_msg_.row_step
        = this->point_cloud_msg_.point_step * this->width;

  sensor_msgs::PointCloud2Modifier pcd_modifier(point_cloud_msg_);
  pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
  pcd_modifier.resize(this->height * this->width);

  // resize if point cloud image to camera parameters if required
  this->point_cloud_image_.create(this->height, this->width, CV_32FC1);

  sensor_msgs::PointCloud2Iterator<float> iter_x(point_cloud_msg_, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(point_cloud_msg_, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(point_cloud_msg_, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(point_cloud_msg_, "rgb");
  cv::MatIterator_<float> iter_image = this->point_cloud_image_.begin<float>();

  point_cloud_msg_.is_dense = true;

  float* toCopyFrom = const_cast<float*>(_src);
  int index = 0;

  double hfov = this->parentSensor->DepthCamera()->HFOV().Radian();
  double fl = static_cast<double>(this->width) / (2.0 * tan(hfov/2.0));

  for (uint32_t j = 0; j < this->height; j++)
  {
    double elevation;
    if (this->height > 1)
      elevation = atan2(static_cast<double>(j) -
                        0.5 * static_cast<double>(this->height-1), fl);
    else
      elevation = 0.0;

    for (uint32_t i = 0; i < this->width;
         i++, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb, ++iter_image)
    {
      double azimuth;
      if (this->width > 1)
        azimuth = atan2(static_cast<double>(i) -
                        0.5 * static_cast<double>(this->width-1), fl);
      else
        azimuth = 0.0;

      double depth = toCopyFrom[index++];

      // in optical frame hardcoded rotation
      // rpy(-M_PI/2, 0, -M_PI/2) is built-in
      // to urdf, where the *_optical_frame should have above relative
      // rotation from the physical camera *_frame
      *iter_x = depth * tan(azimuth);
      *iter_y = depth * tan(elevation);
      if (depth > this->point_cloud_cutoff_)
      {
        *iter_z = depth;
        *iter_image = sqrt(*iter_x * *iter_x +
                           *iter_y * *iter_y +
                           *iter_z * *iter_z);
      }
      else  // point in the unseeable range
      {
        *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
        *iter_image = 0.0;
        point_cloud_msg_.is_dense = false;
      }

      // put image color data for each point
      uint8_t*  image_src = static_cast<uint8_t*>(&(this->image_msg_.data[0]));
      if (this->image_msg_.data.size() == this->height * this->width*3)
      {
        // color
        iter_rgb[0] = image_src[i*3+j*this->width*3+0];
        iter_rgb[1] = image_src[i*3+j*this->width*3+1];
        iter_rgb[2] = image_src[i*3+j*this->width*3+2];
      }
      else if (this->image_msg_.data.size() == this->height * this->width)
      {
        // mono (or bayer?  @todo; fix for bayer)
        iter_rgb[0] = image_src[i+j*this->width];
        iter_rgb[1] = image_src[i+j*this->width];
        iter_rgb[2] = image_src[i+j*this->width];
      }
      else
      {
        // no image
        iter_rgb[0] = 0;
        iter_rgb[1] = 0;
        iter_rgb[2] = 0;
      }
    }
  }
  if (this->point_cloud_connect_count_ > 0)
    this->point_cloud_pub_.publish(this->point_cloud_msg_);

  this->lock_.unlock();
}


/////////////////////////////////////////////////
// incidence angle is target's normal angle accounting for the ray's azimuth
// and elevation
double NpsGazeboRosImageSonar::ComputeIncidence(double azimuth,
                                                double elevation,
                                                cv::Vec3f normal)
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
// Precalculation of corrector sonar calculation
void NpsGazeboRosImageSonar::ComputeCorrector()
{
  double hFOV = this->parentSensor->DepthCamera()->HFOV().Radian();
  double hPixelSize = hFOV / this->width;
  // Beam culling correction precalculation
  for (size_t beam = 0; beam < nBeams; beam ++)
  {
    float beam_azimuthAngle = -(hFOV/2.0) + beam * hPixelSize + hPixelSize/2.0;
    for (size_t beam_other = 0; beam_other < nBeams; beam_other ++)
    {
      float beam_azimuthAngle_other
              = -(hFOV/2.0) + beam_other * hPixelSize + hPixelSize/2.0;
      float azimuthBeamPattern =
        unnormalized_sinc(M_PI * 0.884 / hPixelSize
        * sin(beam_azimuthAngle-beam_azimuthAngle_other));
      this->beamCorrector[beam][beam_other] = azimuthBeamPattern;
      this->beamCorrectorSum += pow(azimuthBeamPattern, 2);
    }
  }
  this->beamCorrectorSum = sqrt(this->beamCorrectorSum);
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
  // cv::dilate(no_readings, no_readings, cv::Mat(),
  //            cv::Point(-1, -1), 2, 1, 1);
  n1.setTo(0, no_readings);
  n2.setTo(0, no_readings);

  std::vector<cv::Mat> images(3);
  cv::Mat white = cv::Mat::ones(depth.rows, depth.cols, CV_32FC1);

  // NOTE: with different focal lengths, the expression becomes
  // (-dzx*fy, -dzy*fx, fx*fy)
  images.at(0) = n1;    // for green channel
  images.at(1) = n2;    // for red channel
  images.at(2) = 1.0/this->focal_length_*depth;  // for blue channel

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
  ROS_DEBUG_NAMED("depth_camera",
    "publishing default camera info, then depth camera info");
  GazeboRosCameraUtils::PublishCameraInfo();

  if (this->depth_info_connect_count_ > 0)
  {
    common::Time sensor_update_time
          = this->parentSensor_->LastMeasurementTime();

    this->sensor_update_time_ = sensor_update_time;
    if (sensor_update_time
          - this->last_depth_image_camera_info_update_time_
          >= this->update_period_)
    {
      this->PublishCameraInfo(this->depth_image_camera_info_pub_);
      // , sensor_update_time);
      this->last_depth_image_camera_info_update_time_ = sensor_update_time;
    }
  }
}

}
