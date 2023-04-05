#include "sv/dsol/node_util.h"

#include "sv/ros2/msg_conv.h"
#include "sv/util/logging.h"

namespace sv::dsol {

namespace gm = geometry_msgs::msg;
namespace vm = visualization_msgs::msg;
static constexpr auto kNaNF = std::numeric_limits<float>::quiet_NaN();

SelectCfg ReadSelectCfg(rclcpp::Node& node, const std::string &ns) {
  SelectCfg cfg;

  std::map<std::string, int> int_params = {
      {"sel_level", cfg.sel_level},
      {"cell_size", cfg.cell_size},
      {"min_grad", cfg.min_grad},
      {"max_grad", cfg.max_grad},
      {"nms_size", cfg.nms_size}
  };
  node.declare_parameters(ns, int_params);

  std::map<std::string, double> double_params = {
      {"min_ratio", cfg.min_ratio},
      {"max_ratio", cfg.max_ratio}
  };
  node.declare_parameters(ns, double_params);

  std::map<std::string, bool> bool_params = {
      {"reselect", cfg.reselect}
  };
  node.declare_parameters(ns, bool_params);

  node.get_parameter(ns + ".sel_level", cfg.sel_level);
  node.get_parameter(ns +".cell_size", cfg.cell_size);
  node.get_parameter(ns +".min_grad", cfg.min_grad);
  node.get_parameter(ns +".select.max_grad", cfg.max_grad);
  node.get_parameter(ns +".nms_size", cfg.nms_size);
  node.get_parameter(ns +".min_ratio", cfg.min_ratio);
  node.get_parameter(ns +".max_ratio", cfg.max_ratio);
  node.get_parameter(ns +".reselect", cfg.reselect);
  return cfg;
}

DirectCfg ReadDirectCfg(rclcpp::Node& node, const std::string &ns) {
  DirectCfg cfg;

  std::map<std::string, int> int_params = {
      {"init_level", cfg.optm.init_level},
      {"max_iters", cfg.optm.max_iters},
      {"c2", cfg.cost.c2},
      {"dof", cfg.cost.dof},
      {"max_outliers", cfg.cost.max_outliers}
  };
  node.declare_parameters(ns, int_params);

  std::map<std::string, double> double_params = {
      {"max", cfg.optm.max_xs},
      {"grad_factor", cfg.cost.grad_factor},
      {"min_depth", cfg.cost.min_depth}
  };
  node.declare_parameters(ns, double_params);

  std::map<std::string, bool> bool_params = {
      {"stereo", cfg.cost.stereo},
      {"affine", cfg.cost.affine}
  };
  node.declare_parameters(ns, bool_params);

  node.get_parameter(ns + ".init_level", cfg.optm.init_level);
  node.get_parameter(ns + ".max_iters", cfg.optm.max_iters);
  node.get_parameter(ns + ".max_xs", cfg.optm.max_xs);

  node.get_parameter(ns + ".affine", cfg.cost.affine);
  node.get_parameter(ns + ".stereo", cfg.cost.stereo);
  node.get_parameter(ns + ".c2", cfg.cost.c2);
  node.get_parameter(ns + ".dof", cfg.cost.dof);
  node.get_parameter(ns + ".max_outliers", cfg.cost.max_outliers);
  node.get_parameter(ns + ".grad_factor", cfg.cost.grad_factor);
  node.get_parameter(ns + ".min_depth", cfg.cost.min_depth);

  return cfg;
}

StereoCfg ReadStereoCfg(rclcpp::Node& node, const std::string &ns) {
  StereoCfg cfg;

  std::map<std::string, int> int_params = {
      {"half_rows", cfg.half_rows},
      {"half_cols", cfg.half_cols},
      {"match_level", cfg.match_level},
      {"refine_size", cfg.refine_size}
  };
  node.declare_parameters(ns, int_params);

  std::map<std::string, double> double_params = {
      {"min_zncc", cfg.min_zncc},
      {"min_depth", cfg.min_depth}
  };
  node.declare_parameters(ns, double_params);

  node.get_parameter(ns + ".half_rows", cfg.half_rows);
  node.get_parameter(ns + ".half_cols", cfg.half_cols);
  node.get_parameter(ns + ".match_level", cfg.match_level);
  node.get_parameter(ns + ".refine_size", cfg.refine_size);
  node.get_parameter(ns + ".min_zncc", cfg.min_zncc);
  node.get_parameter(ns + ".min_depth", cfg.min_depth);
  return cfg;
}

OdomCfg ReadOdomCfg(rclcpp::Node& node, const std::string &ns) {
  OdomCfg cfg;

  std::map<std::string, int> int_params = {
      {"num_kfs", cfg.num_kfs},
      {"num_levels", cfg.num_levels}
  };
  node.declare_parameters(ns, int_params);

  std::map<std::string, double> double_params = {
      {"min_track_ratio", cfg.min_track_ratio},
      {"vis_min_depth", cfg.vis_min_depth}
  };
  node.declare_parameters(ns, double_params);

  std::map<std::string, bool> bool_params = {
      {"marg", cfg.marg},
      {"reinit", cfg.reinit},
      {"init_depth", cfg.init_depth},
      {"init_stereo", cfg.init_stereo},
      {"init_align", cfg.init_align}
  };
  node.declare_parameters(ns, bool_params);

  node.get_parameter(ns + ".marg", cfg.marg);
  node.get_parameter(ns + ".num_kfs", cfg.num_kfs);
  node.get_parameter(ns + ".num_levels", cfg.num_levels);
  node.get_parameter(ns + ".min_track_ratio", cfg.min_track_ratio);
  node.get_parameter(ns + ".vis_min_depth", cfg.vis_min_depth);

  node.get_parameter(ns + ".reinit", cfg.reinit);
  node.get_parameter(ns + ".init_depth", cfg.init_depth);
  node.get_parameter(ns + ".init_stereo", cfg.init_stereo);
  node.get_parameter(ns + ".init_align", cfg.init_align);
  return cfg;
}

Camera MakeCamera(const sensor_msgs::msg::CameraInfo& cinfo_msg) {
  const cv::Size size(cinfo_msg.width, cinfo_msg.height);
  const auto& P = cinfo_msg.p;
  CHECK_GT(P[0], 0);
  // P
  // 0, 1,  2,  3
  // 4, 5,  6,  7
  // 8, 9, 10, 11
  Eigen::Array4d fc;
  fc << P[0], P[5], P[2], P[6];
  return {size, fc, -P[3] / P[0]};
}

void Keyframe2Cloud(const Keyframe& keyframe,
                    sensor_msgs::msg::PointCloud2& cloud,
                    double max_depth,
                    int offset) {
  const auto& points = keyframe.points();
  const auto& patches = keyframe.patches().front();
  const auto grid_size = points.cvsize();

  const auto total_size = offset + grid_size.area();
  cloud.data.resize(total_size * cloud.point_step);
  cloud.height = 1;
  cloud.width = total_size;

  for (int gr = 0; gr < points.rows(); ++gr) {
    for (int gc = 0; gc < points.cols(); ++gc) {
      const auto i = offset + gr * grid_size.width + gc;
      auto* ptr =
          reinterpret_cast<float*>(cloud.data.data() + i * cloud.point_step);

      const auto& point = points.at(gr, gc);
      // Only draw points with max info and within max depth
      if (!point.InfoMax() || (1.0 / point.idepth()) > max_depth) {
        ptr[0] = ptr[1] = ptr[2] = kNaNF;
        continue;
      }
      CHECK(point.PixelOk());
      CHECK(point.DepthOk());

      // transform to fixed frame
      const Eigen::Vector3f p_w = (keyframe.Twc() * point.pt()).cast<float>();
      const auto& patch = patches.at(gr, gc);

      ptr[0] = p_w.x();
      ptr[1] = p_w.y();
      ptr[2] = p_w.z();
      ptr[3] = static_cast<float>(patch.vals[0] / 255.0);
    }
  }
}

void Keyframes2Cloud(const KeyframePtrConstSpan& keyframes,
                     sensor_msgs::msg::PointCloud2& cloud,
                     double max_depth) {
  if (keyframes.empty()) return;

  const auto num_kfs = static_cast<int>(keyframes.size());
  const auto grid_size = keyframes[0]->points().cvsize();

  // Set all points to bad
  const auto total_size = num_kfs * grid_size.area();
  cloud.data.reserve(total_size * cloud.point_step);
  cloud.height = 1;
  cloud.width = total_size;

  for (int k = 0; k < num_kfs; ++k) {
    Keyframe2Cloud(
        GetKfAt(keyframes, k), cloud, max_depth, grid_size.area() * k);
  }
}

/// ============================================================================

void DrawAlignGraph(const Eigen::Vector3d& frame_pos,
                    const Eigen::Matrix3Xd& kfs_pos,
                    const std::vector<int>& tracks,
                    const cv::Scalar& color,
                    double scale,
                    vm::Marker& marker) {
  CHECK_EQ(tracks.size(), kfs_pos.cols());
  marker.ns = "align";
  marker.id = 0;
  marker.type = vm::Marker::LINE_LIST;
  marker.action = vm::Marker::ADD;
  marker.color.b = static_cast<float>(color[0]);
  marker.color.g = static_cast<float>(color[1]);
  marker.color.r = static_cast<float>(color[2]);
  marker.color.a = 1.0F;

  marker.scale.x = scale;
  marker.pose.orientation.w = 1.0;
  const auto num_kfs = tracks.size();
  marker.points.clear();
  marker.points.reserve(num_kfs * 2);

  gm::Point p0;
  p0.x = frame_pos.x();
  p0.y = frame_pos.y();
  p0.z = frame_pos.z();

  gm::Point p1;
  for (long unsigned i = 0; i < num_kfs; ++i) {
    if (tracks[i] <= 0) continue;
    p1.x = kfs_pos.col(i).x();
    p1.y = kfs_pos.col(i).y();
    p1.z = kfs_pos.col(i).z();
    marker.points.push_back(p0);
    marker.points.push_back(p1);
  }
}

PosePathPublisher::PosePathPublisher(rclcpp::Node& pnh,
                                     const std::string& name,
                                     const std::string& frame_id)
    : frame_id_{frame_id},
      pose_pub_{pnh.create_publisher<gm::PoseStamped>("pose_" + name, 1)},
      path_pub_{pnh.create_publisher<nav_msgs::msg::Path>("path_" + name, 1)} {
  path_msg_.poses.reserve(1024);
}

gm::PoseStamped PosePathPublisher::Publish(const rclcpp::Time& time,
                                           const Sophus::SE3d& tf) {
  gm::PoseStamped pose_msg;
  pose_msg.header.stamp = time;
  pose_msg.header.frame_id = frame_id_;
  Sophus2Ros(tf, pose_msg.pose);
  pose_pub_->publish(pose_msg);

  path_msg_.header = pose_msg.header;
  path_msg_.poses.push_back(pose_msg);
  path_pub_->publish(path_msg_);
  return pose_msg;
}

}  // namespace sv::dsol
