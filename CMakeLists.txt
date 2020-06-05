cmake_minimum_required(VERSION 2.8.3)
project(nps_uw_sensors_gazebo)

find_package(catkin REQUIRED tf)

find_package(gazebo REQUIRED)
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV)
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${GAZEBO_INCLUDE_DIRS}
)
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

catkin_package()

## Plugins
add_library(gazebo_ros_pulse_lidar_plugin src/gazebo_ros_pulse_lidar.cpp)
add_library(nps_gazebo_ros_gpu_sonar_single_beam_plugin src/nps_gazebo_ros_gpu_sonar_single_beam.cpp)
target_link_libraries(gazebo_ros_pulse_lidar_plugin nps_gazebo_ros_gpu_sonar_single_beam_plugin
                      GpuRayPlugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})

# for launch
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)

