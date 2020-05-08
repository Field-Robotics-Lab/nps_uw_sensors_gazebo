cmake_minimum_required(VERSION 2.8.3)
project(nps_uw_sensors)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  pcl_ros
  cv_bridge
  image_transport
  tf
)

find_package(gazebo REQUIRED)
find_package(OpenCV)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR} 
  ${catkin_INCLUDE_DIRS} 
  ${GAZEBO_INCLUDE_DIRS}
)
link_directories(${GAZEBO_LIBRARY_DIRS})

catkin_package(
LIBRARIES
INCLUDE_DIRS
DEPENDS
)

## Plugins
add_library(nps_gazebo_ros_gpu_sonar_single_beam_plugin
            src/nps_gazebo_ros_gpu_sonar_single_beam.cpp)
target_link_libraries(nps_gazebo_ros_gpu_sonar_single_beam_plugin
                      GpuRayPlugin ${catkin_LIBRARIES})

# for launch
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)

