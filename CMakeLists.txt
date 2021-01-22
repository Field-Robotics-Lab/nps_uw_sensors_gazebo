cmake_minimum_required(VERSION 3.0.2)
project(nps_uw_sensors_gazebo)

if(NOT "${CMAKE_VERSION}" VERSION_LESS "3.16")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
    add_compile_options(-std=c++11)
endif()

find_package(catkin REQUIRED COMPONENTS
 tf
 gazebo_plugins)

find_package(gazebo REQUIRED)
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")
set(SENSOR_ROS_PLUGINS_LIST "")

catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS
 )

## Plugins

# Install plugins
install(
  TARGETS ${SENSOR_ROS_PLUGINS_LIST}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# for launch
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)

# for Python scripts
catkin_install_python(PROGRAMS
  src/simple_box_motion.py
  src/simple_motion.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
