#pragma once

#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cv_bridge/cv_bridge.h>
//#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace sv {

// Vector3d <-> Point
void Eigen2Ros(const Eigen::Vector3d& e, geometry_msgs::msg::Point& r);
void Ros2Eigen(const geometry_msgs::msg::Point& r, Eigen::Ref<Eigen::Vector3d> e);

// Vector3d <-> Vector3
void Eigen2Ros(const Eigen::Vector3d& e, geometry_msgs::msg::Vector3& r);
void Ros2Eigen(const geometry_msgs::msg::Vector3& r, Eigen::Ref<Eigen::Vector3d> e);

// Quateriond <-> Quaternion
void Eigen2Ros(const Eigen::Quaterniond& e, geometry_msgs::msg::Quaternion& r);
void Ros2Eigen(const geometry_msgs::msg::Quaternion& r, Eigen::Quaterniond& e);

// Vector3d, Quaterniond <-> Pose / Transform
void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               geometry_msgs::msg::Pose& pose);
void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               geometry_msgs::msg::Transform& tf);

// Isometry3d <-> Transform / Pose
void Ros2Eigen(const geometry_msgs::msg::Transform& tf, Eigen::Isometry3d& iso);
void Eigen2Ros(const Eigen::Isometry3d& iso, geometry_msgs::msg::Pose& pose);

void Sophus2Ros(const Sophus::SE3d& se3, geometry_msgs::msg::Pose& pose);
void Sophus2Ros(const Sophus::SE3d& se3, geometry_msgs::msg::Transform& tf);
void Sophus2Ros(const Sophus::SO3d& so3, geometry_msgs::msg::Quaternion& quat);
void Ros2Sophus(const geometry_msgs::msg::Quaternion& quat, Sophus::SO3d& so3);
void Ros2Sophus(const geometry_msgs::msg::Pose& pose, Sophus::SE3d& se3);

void Ros2Ros(const geometry_msgs::msg::Pose& pose, geometry_msgs::msg::Transform& tf);

using PointFields = std::vector<sensor_msgs::msg::PointField>;

PointFields MakePointFields(const std::string& fstr);
int GetPointStep(const PointFields& fields);

/// @brief A simple wrapper around PointCloud2
struct Cloud2Helper {
  sensor_msgs::msg::PointCloud2 cloud;

  auto width() const noexcept { return cloud.width; }
  auto height() const noexcept { return cloud.height; }
  auto rows() const noexcept { return height(); }
  auto cols() const noexcept { return width(); }
  auto empty() const noexcept { return cloud.data.empty(); }

  Cloud2Helper() = default;
  Cloud2Helper(int rows, int cols, const PointFields& fields);
  Cloud2Helper(int rows, int cols, const std::string& fstring)
      : Cloud2Helper(rows, cols, MakePointFields(fstring)) {}
  explicit Cloud2Helper(const std::string& fstring)
      : Cloud2Helper(0, 0, fstring) {}

  void resize(int rows, int cols);

  template <typename T>
  T* PtrAt(int i) {
    return reinterpret_cast<T*>(cloud.data.data() + i * cloud.point_step);
  }

  template <typename T>
  T* PtrAt(int r, int c) {
    return PtrAt<T>(r * cloud.width + c);
  }
};

}  // namespace sv
