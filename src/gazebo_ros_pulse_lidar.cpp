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

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <string>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32.h"

namespace gazebo
{
  /// \brief A plugin to control a Velodyne sensor.
  class GazeboRosPulseLidar : public ModelPlugin
  {
    /// \brief Constructor
    public: GazeboRosPulseLidar() {}

    /// \brief The load function is called by Gazebo when the plugin is
    /// inserted into simulation
    /// \param[in] _model A pointer to the model that this plugin is
    /// attached to.
    /// \param[in] _sdf A pointer to the plugin's SDF element.
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
     // Safety check
     if (_model->GetJointCount() == 0)
     {
       ROS_FATAL_STREAM_NAMED("pulse_lidar", "Invalid joint count; "
                           "NPS Gazebo ROS lidar plugin will not be loaded.");
       return;
     }
      ROS_INFO_NAMED("pulse_lidar", 
                     "The NPS Gazebo ROS pulse lidar plugin is attached to "
                     "model [%s]", _model->GetName().c_str());
 
     // Store the model pointer for convenience.
     this->model = _model;

     // Get the joints
     this->pan_joint = this->model->GetJoint("3dad_sl3::base_top_joint");
     this->tilt_joint = this->model->GetJoint("3dad_sl3::top_tray_joint");

     // Setup a P-controller, with _imax = 1
     this->pan_pid = common::PID(1, 0, 2.5, 1);

     // Setup a P-controller with _imax = 10
     // This is very unstable out of water and requires very high gain values
     // to get close to compliance.
     this->tilt_pid = common::PID(250, 50, 100, 50);

     // Apply the P0-controller to the joint.
     this->model->GetJointController()->SetPositionPID(
         this->pan_joint->GetScopedName(), this->pan_pid);

     this->model->GetJointController()->SetPositionPID(
         this->tilt_joint->GetScopedName(), this->tilt_pid);


     // Default to zero velocity
     double pan_position = 0;
     double tilt_position = 0;

     // Check that the velocity element exists, then read the value
     if (_sdf->HasElement("pan_position"))
       pan_position = _sdf->Get<double>("pan_position");

     // Check that the velocity element exists, then read the value
     if (_sdf->HasElement("tilt_position"))
       tilt_position = _sdf->Get<double>("tilt_position");

     // Set the joint's target velocity. This target velocity is just
     // for demonstration purposes.
     this->model->GetJointController()->SetPositionTarget(
         this->pan_joint->GetScopedName(), pan_position);

     this->model->GetJointController()->SetPositionTarget(
         this->tilt_joint->GetScopedName(), tilt_position);

     // Create the node
     this->node = transport::NodePtr(new transport::Node());
     #if GAZEBO_MAJOR_VERSION < 8
     this->node->Init(this->model->GetWorld()->GetName());
     #else
     this->node->Init(this->model->GetWorld()->Name());
     #endif

     // Create a topic name
     std::string topicName = "~/" + this->model->GetName() + "/lidar_cmd";

     // Subscribe to the topic, and register a callback
     this->sub = this->node->Subscribe(topicName,
        &GazeboRosPulseLidar::OnMsg, this);

     // Initialize ros, if it has not already bee initialized.
     if (!ros::isInitialized())
     {
       int argc = 0;
       char **argv = NULL;
       ros::init(argc, argv, "gazebo_client",
           ros::init_options::NoSigintHandler);
     }

     // Create our ROS node. This acts in a similar manner to
     // the Gazebo node
     this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

     // Create a named topic, and subscribe to it.
     ros::SubscribeOptions so_pan =
       ros::SubscribeOptions::create<std_msgs::Float32>(
           "/" + this->model->GetName() + "/lidar_pan_cmd",
           1,
           boost::bind(&GazeboRosPulseLidar::OnRosPanMsg, this, _1),
           ros::VoidPtr(), &this->rosQueue);
     this->rosSubPan = this->rosNode->subscribe(so_pan);

     this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

     // Create a named topic, and subscribe to it.
     ros::SubscribeOptions so_tilt =
       ros::SubscribeOptions::create<std_msgs::Float32>(
           "/" + this->model->GetName() + "/lidar_tilt_cmd",
           1,
           boost::bind(&GazeboRosPulseLidar::OnRosTiltMsg, this, _1),
           ros::VoidPtr(), &this->rosQueue);
     this->rosSubTilt = this->rosNode->subscribe(so_tilt);

     // Spin up the queue helper thread.
     this->rosQueueThread =
       std::thread(std::bind(&GazeboRosPulseLidar::QueueThread, this));
    }

    /// \brief Set the position of the lidar
    /// \param[in] _vel New target position
    public: void SetPanPosition(const double &_pos)
    {
      // Set the joint's target velocity.
      this->model->GetJointController()->SetPositionTarget(
          this->pan_joint->GetScopedName(), _pos);
    }

    /// \brief Set the position of the lidar
    /// \param[in] _vel New target position
    public: void SetTiltPosition(const double &_pos)
    {
      // Set the joint's target velocity.
      this->model->GetJointController()->SetPositionTarget(
          this->tilt_joint->GetScopedName(), _pos);
    }


    /// \brief Handle incoming message
    /// \param[in] _msg Repurpose a vector3 message. This function will
    /// only use the x component.
    private: void OnMsg(ConstVector3dPtr &_msg)
    {
      this->SetPanPosition(_msg->x());
      this->SetTiltPosition(_msg->y());
    }

    /// \brief Handle an incoming message from ROS
    /// \param[in] _msg A float value that is used to set the velocity
    /// of the lidar.
    public: void OnRosPanMsg(const std_msgs::Float32ConstPtr &_msg)
    {
      this->SetPanPosition(_msg->data);
    }

    public: void OnRosTiltMsg(const std_msgs::Float32ConstPtr &_msg)
    {
      this->SetTiltPosition(_msg->data);
    }

    /// \brief ROS helper function that processes messages
    private: void QueueThread()
    {
      static const double timeout = 0.01;
      while (this->rosNode->ok())
      {
        this->rosQueue.callAvailable(ros::WallDuration(timeout));
      }
    }

    /// \brief A node used for transport
    private: transport::NodePtr node;

    /// \brief A subscriber to a named topic.
    private: transport::SubscriberPtr sub;

    public: physics::JointControllerPtr jointController;

    /// \brief Pointer to the model.
    private: physics::ModelPtr model;

    /// \brief Pointer to the joint.
    private: physics::JointPtr pan_joint;

    /// \brief Pointer to the joint.
    private: physics::JointPtr tilt_joint;

     /// \brief A PID controller for pan and tilt motion.
    private: common::PID pan_pid;

    /// \brief A PID controller for pan and tilt motion.
    private: common::PID tilt_pid;

    /// \brief A node use for ROS transport
    private: std::unique_ptr<ros::NodeHandle> rosNode;

    /// \brief A ROS subscriber
    private: ros::Subscriber rosSubPan;

    /// \brief A ROS subscriber
    private: ros::Subscriber rosSubTilt;

    /// \brief A ROS callbackqueue that helps process messages
    private: ros::CallbackQueue rosQueue;

    /// \brief A thread the keeps running the rosQueue
    private: std::thread rosQueueThread;
  };

  // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
  GZ_REGISTER_MODEL_PLUGIN(GazeboRosPulseLidar)
}
