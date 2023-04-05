cmake_minimum_required(VERSION 3.15)
project(dsol)

CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_ROS "Build ros" ON)
option(BUILD_PLAY "Build play" OFF)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_VECTORIZE "Build with -march=native" ON)

set(CC_TARGET_PREFIX sv)
include(CMakeHelpers)

add_library(sv_options INTERFACE)
include(CompilerWarnings)
enable_warnings(sv_options)
include(Sanitizers)
enable_sanitizers(sv_options)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

#find_package(
#  catkin QUIET
#  COMPONENTS roscpp
#             tf2_ros
#             tf2_eigen
#             cv_bridge
#             image_transport
#             nav_msgs
#             sensor_msgs
#             visualization_msgs)

find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(absl REQUIRED)
find_package(Boost REQUIRED) # circular buffer

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xtensor) # reading npy

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

if(NOT ament_cmake_FOUND)
  set(BUILD_ROS OFF)
  message(WARN "BUILD_ROS set to OFF because ament not found.")
endif()

if(BUILD_ROS)
#  catkin_package()
#  ament_target_dependencies(dsso_ros rclcpp sensor_msgs geometry_msgs cv_bridge)
endif()

enable_testing()
add_subdirectory(sv)

if(BUILD_ROS)
  install(DIRECTORY
          launch config
          DESTINATION share/${PROJECT_NAME}/
          )
endif()

ament_package()