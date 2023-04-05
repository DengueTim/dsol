#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "sv/dsol/direct.h"
#include "sv/dsol/odom.h"
#include "sv/dsol/select.h"
#include "sv/dsol/stereo.h"

namespace sv::dsol {

SelectCfg ReadSelectCfg(rclcpp::Node& node, const std::string &ns);
DirectCfg ReadDirectCfg(rclcpp::Node& node, const std::string &ns);
StereoCfg ReadStereoCfg(rclcpp::Node& node, const std::string &ns);
OdomCfg ReadOdomCfg(rclcpp::Node& node, const std::string &ns);

Camera MakeCamera(const sensor_msgs::msg::CameraInfo& cinfo_msg);

void Keyframe2Cloud(const Keyframe& kefyrame,
                    sensor_msgs::msg::PointCloud2& cloud,
                    double max_depth,
                    int offset = 0);
void Keyframes2Cloud(const KeyframePtrConstSpan& keyframes,
                     sensor_msgs::msg::PointCloud2& cloud,
                     double max_depth);

void DrawAlignGraph(const Eigen::Vector3d& frame_pos,
                    const Eigen::Matrix3Xd& kfs_pos,
                    const std::vector<int>& tracks,
                    const cv::Scalar& color,
                    double scale,
                    visualization_msgs::msg::Marker& marker);

/// @brief Publish pose and also add it to path
struct PosePathPublisher {
  PosePathPublisher() = default;
  PosePathPublisher(rclcpp::Node& pnh,
                    const std::string& name,
                    const std::string& frame_id);

  geometry_msgs::msg::PoseStamped Publish(const rclcpp::Time& time,
                                     const Sophus::SE3d& tf);

  std::string frame_id_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  nav_msgs::msg::Path path_msg_;
};

}  // namespace sv::dsol
