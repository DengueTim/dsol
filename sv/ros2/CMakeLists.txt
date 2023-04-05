
#cc_library(
#  NAME ros2
#  DEPS rclcpp sensor_msgs geometry_msgs cv_bridge
#  INCS ${geometry_msgs_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS} ${cv_bridge_INCLUDE_DIRS}
#  INTERFACE)

#add_library(sv_ros2 INTERFACE)
#target_include_directories(sv_ros2 INTERFACE ${geometry_msgs_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS} ${cv_bridge_INCLUDE_DIRS})
#target_link_libraries(sv_ros2 INTERFACE rclcpp sensor_msgs geometry_msgs cv_bridge)
#target_compile_definitions(sv_ros2 INTERFACE ${CC_LIB_DEFINES})
#target_compile_options(sv_ros2 INTERFACE ${CC_LIB_COPTS})
#ament_target_dependencies(sv_ros2  rclcpp tf2 tf2_ros sensor_msgs geometry_msgs nav_msgs cv_bridge)

#
#cc_library(
#  NAME ros2_msg_conv
#  SRCS "msg_conv.cpp"
#  DEPS sv_base sv_ros2 Eigen3::Eigen Sophus::Sophus)

add_library(sv_ros2_msg_conv)
#target_include_directories(sv_ros2_msg_conv INTERFACE ${geometry_msgs_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS} ${cv_bridge_INCLUDE_DIRS})
target_sources(sv_ros2_msg_conv PRIVATE msg_conv.cpp)
#target_link_libraries(sv_ros2_msg_conv PUBLIC sv_base sv_ros2 Eigen3::Eigen Sophus::Sophus)
ament_target_dependencies(sv_ros2_msg_conv rclcpp tf2 tf2_ros sensor_msgs geometry_msgs nav_msgs visualization_msgs cv_bridge)

include_directories(
        ${PROJECT_SOURCE_DIR}
#        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
#        ${Boost_INCLUDE_DIR}
#        ${CSPARSE_INCLUDE_DIR}
)

install(TARGETS
        sv_ros2_msg_conv
        DESTINATION lib/${PROJECT_NAME})
