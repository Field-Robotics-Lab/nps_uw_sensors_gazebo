cmake_minimum_required(VERSION 2.8.3)
project(nps_uw_sensors_gazebo)

## Compile as C++11, supported in ROS Kinetic and newer
# If running on focal, will be using Noetic + Gazebo11, which requires c++17
execute_process(COMMAND lsb_release -cs
  OUTPUT_VARIABLE RELEASE_CODENAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(RELEASE_CODENAME MATCHES "focal")
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  add_compile_options(-std=c++11)
endif()

find_package(catkin REQUIRED tf gazebo_plugins)

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
target_link_libraries(gazebo_ros_pulse_lidar_plugin
                      ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})

add_library(nps_gazebo_ros_gpu_sonar_single_beam_plugin
            src/nps_gazebo_ros_gpu_sonar_single_beam.cpp)
target_link_libraries(nps_gazebo_ros_gpu_sonar_single_beam_plugin
                      GpuRayPlugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(nps_gazebo_ros_depth_camera_sonar_single_beam_plugin
            src/nps_gazebo_ros_depth_camera_sonar_single_beam.cpp
            include/nps_uw_sensors_gazebo/nps_gazebo_ros_depth_camera_sonar_single_beam.hh)
target_link_libraries(nps_gazebo_ros_depth_camera_sonar_single_beam_plugin
                      DepthCameraPlugin ${catkin_LIBRARIES})

# for launch
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)

# for Python scripts
catkin_install_python(PROGRAMS scripts/depth_camera_sonar.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
