#include "rclcpp/rclcpp.hpp"
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <boost/circular_buffer.hpp>

#include "sv/dsol/extra.h"
#include "sv/dsol/node_util.h"
#include "sv/dsol/odom.h"
#include "sv/ros2/msg_conv.h"

namespace sv::dsol {

namespace cb = cv_bridge;
namespace sm = sensor_msgs::msg;
namespace gm = geometry_msgs::msg;
namespace mf = message_filters;

class NodeOdom : public rclcpp::Node {
 public:
  NodeOdom();

  void InitOdom();
  void InitRosIO();

  void Cinfo1Cb(const sm::CameraInfo& cinfo1_msg);
  void StereoCb(const sm::Image::ConstSharedPtr& image0_ptr,
                const sm::Image::ConstSharedPtr& image1_ptr);
  void StereoDepthCb(const sm::Image::ConstSharedPtr& image0_ptr,
                     const sm::Image::ConstSharedPtr& image1_ptr,
                     const sm::Image::ConstSharedPtr& depth0_ptr);

  void TfCamCb(const gm::Transform& tf_cam_msg);
  void TfImuCb(const gm::Transform& tf_imu_msg);

  void AccCb(const sm::Imu& acc_msg);
  void GyrCb(const sm::Imu& gyr_msg);

  void PublishOdom(const std_msgs::msg::Header& header, const Sophus::SE3d& tf);
  void PublishCloud(const std_msgs::msg::Header& header);

  using SyncStereo = mf::TimeSynchronizer<sm::Image, sm::Image>;
  using SyncStereoDepth = mf::TimeSynchronizer<sm::Image, sm::Image, sm::Image>;

  boost::circular_buffer<sm::Imu> gyrs_;
  mf::Subscriber<sm::Image> sub_image0_;
  mf::Subscriber<sm::Image> sub_image1_;
  mf::Subscriber<sm::Image> sub_depth0_;

  std::optional<SyncStereo> sync_stereo_;
  std::optional<SyncStereoDepth> sync_stereo_depth_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cinfo1_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_acc_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_gyr_;

  rclcpp::Publisher<sm::PointCloud2>::SharedPtr pub_points_;
  rclcpp::Publisher<gm::PoseArray>::SharedPtr pub_parray_;
  PosePathPublisher pub_odom_;

  MotionModel motion_;
  DirectOdometry odom_;

  std::string frame_{"fixed"};
  sm::PointCloud2 cloud_;
};

NodeOdom::NodeOdom() : Node("dsol_odom"),
      gyrs_(50),
      sub_image0_(this, "image0", rmw_qos_profile_sensor_data),
      sub_image1_(this, "image1", rmw_qos_profile_sensor_data),
      sub_depth0_(this, "depth0", rmw_qos_profile_sensor_data) {
  InitOdom();
  InitRosIO();
}

void NodeOdom::InitOdom() {
  {
    auto cfg = ReadOdomCfg(*this, "odom");

    declare_parameter<int>("tbb");
    declare_parameter<int>("log");
    declare_parameter<int>("vis");

    get_parameter("tbb", cfg.tbb);
    get_parameter("log", cfg.log);
    get_parameter("vis", cfg.vis);
    odom_.Init(cfg);
  }
  odom_.selector = PixelSelector(ReadSelectCfg(*this, "select"));
  odom_.matcher = StereoMatcher(ReadStereoCfg(*this, "stereo"));
  odom_.aligner = FrameAligner(ReadDirectCfg(*this, "align"));
  odom_.adjuster = BundleAdjuster(ReadDirectCfg(*this, "adjust"));
  odom_.cmap = GetColorMap(get_parameter_or<std::string>("cm", "jet"));
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), odom_.Repr());

  // Init motion model
  motion_.Init();
}

void NodeOdom::InitRosIO() {
  declare_parameter<bool>("use_depth");
  bool use_depth = get_parameter_or("use_depth", false);
  if (use_depth) {
    sync_stereo_depth_.emplace(sub_image0_, sub_image1_, sub_depth0_, 5);
    sync_stereo_depth_->registerCallback(
        std::bind(&NodeOdom::StereoDepthCb, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  } else {
    sync_stereo_.emplace(sub_image0_, sub_image1_, 5);
    sync_stereo_->registerCallback(
        std::bind(&NodeOdom::StereoCb, this, std::placeholders::_1, std::placeholders::_2));
  }

  std::function<void(sensor_msgs::msg::CameraInfo)> fnc1;
  fnc1 = std::bind(&NodeOdom::Cinfo1Cb, this, std::placeholders::_1);
  sub_cinfo1_ = create_subscription<sensor_msgs::msg::CameraInfo>("cinfo1", 1, fnc1);

  std::function<void(sensor_msgs::msg::Imu)> fnc2;
  fnc2 = std::bind(&NodeOdom::GyrCb, this, std::placeholders::_1);
  sub_gyr_ = create_subscription<sensor_msgs::msg::Imu>("gyr", 200, fnc2);

  std::function<void(sensor_msgs::msg::Imu)> fnc3;
  fnc3 = std::bind(&NodeOdom::AccCb, this, std::placeholders::_1);
  sub_acc_ = create_subscription<sensor_msgs::msg::Imu>("acc", 100, fnc3);

  pub_odom_ = PosePathPublisher(*this, "odom", frame_);
  pub_points_ = create_publisher<sm::PointCloud2>("points", 1);
  pub_parray_ = create_publisher<gm::PoseArray>("parray", 1);
}

void NodeOdom::Cinfo1Cb(const sensor_msgs::msg::CameraInfo& cinfo1_msg) {
  odom_.camera = MakeCamera(cinfo1_msg);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), odom_.camera.Repr());
  //sub_cinfo1_->shutdown();
}

void NodeOdom::AccCb(const sensor_msgs::msg::Imu& acc_msg) {}

void NodeOdom::GyrCb(const sensor_msgs::msg::Imu& gyr_msg) {
  // Normally there is a transform from imu to camera, but in realsense, imu and
  // left infrared camera are aligned (only small translation, so we skip
  // reading the tf)

  gyrs_.push_back(gyr_msg);
}

void NodeOdom::StereoCb(const sensor_msgs::msg::Image::ConstSharedPtr& image0_ptr,
                        const sensor_msgs::msg::Image::ConstSharedPtr& image1_ptr) {
  StereoDepthCb(image0_ptr, image1_ptr, nullptr);
}

void NodeOdom::StereoDepthCb(const sensor_msgs::msg::Image::ConstSharedPtr& image0_ptr,
                             const sensor_msgs::msg::Image::ConstSharedPtr& image1_ptr,
                             const sensor_msgs::msg::Image::ConstSharedPtr& depth0_ptr) {
  const auto curr_header = image0_ptr->header;
  const auto image0 = cb::toCvShare(image0_ptr)->image;
  const auto image1 = cb::toCvShare(image1_ptr)->image;

  // depth
  cv::Mat depth0;
  if (depth0_ptr) {
    depth0 = cb::toCvCopy(depth0_ptr)->image;
    depth0.convertTo(depth0, CV_32FC1, 0.001);  // 16bit in millimeters
  }

  // Get delta time
  static rclcpp::Time prev_stamp(0);
  const rclcpp::Duration delta_duration =
      prev_stamp == rclcpp::Time(0) ? rclcpp::Duration(0, 0) : rclcpp::Time(curr_header.stamp) - prev_stamp;
  const auto dt = delta_duration.seconds();
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "dt: " << dt * 1000);

  // Motion model
  Sophus::SE3d dtf_pred;
  if (dt > 0) {
    // Do a const vel prediction first
    dtf_pred = motion_.PredictDelta(dt);

    // Then overwrite rotation part if we have imu
    // TODO(dsol): Use 0th order integration, maybe switch to 1st order later
    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),
                       fmt::format("prev: {}, curr: {}, first_imu: {}, last_imu: {}",
                    prev_stamp.seconds(),
                    curr_header.stamp.sec,
                    gyrs_.front().header.stamp.sec,
                    gyrs_.back().header.stamp.sec));
    Sophus::SO3d dR{};
    int n_imus = 0;
    for (size_t i = 0; i < gyrs_.size(); ++i) {
      const auto& imu = gyrs_[i];
      // Skip imu msg that is earlier than the previous odom
      if (rclcpp::Time(imu.header.stamp) <= prev_stamp) continue;
      if (rclcpp::Time(imu.header.stamp) > curr_header.stamp) continue;

      const auto prev_imu_stamp =
          i == 0 ? prev_stamp : rclcpp::Time(gyrs_.at(i - 1).header.stamp);
      const double dt_imu = (rclcpp::Time(imu.header.stamp) - prev_imu_stamp).seconds();
      CHECK_GT(dt_imu, 0);
      Eigen::Map<const Eigen::Vector3d> w(&imu.angular_velocity.x);
      dR *= Sophus::SO3d::exp(w * dt_imu);
      ++n_imus;
    }
    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "n_imus: " << n_imus);
    // We just replace const vel prediction
    if (n_imus > 0) dtf_pred.so3() = dR;
  }

  const auto status = odom_.Estimate(image0, image1, dtf_pred, depth0);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), status.Repr());

  // Motion model correct if tracking is ok and not first frame
  if (status.track.ok) {
    motion_.Correct(status.Twc(), dt);
  } else {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("rclcpp"), "Tracking failed (or 1st frame), slow motion model");
  }

  // publish stuff
  std_msgs::msg::Header header;
  header.frame_id = "fixed";
  header.stamp = curr_header.stamp;

  PublishOdom(header, status.Twc());
  if (status.map.remove_kf) {
    PublishCloud(header);
  }

  prev_stamp = curr_header.stamp;
}

void NodeOdom::PublishOdom(const std_msgs::msg::Header& header,
                           const Sophus::SE3d& tf) {
  // Publish odom poses
  const auto pose_msg = pub_odom_.Publish(header.stamp, tf);

  // Publish keyframe poses
  const auto poses = odom_.window.GetAllPoses();
  gm::PoseArray parray_msg;
  parray_msg.header = header;
  parray_msg.poses.resize(poses.size());
  for (size_t i = 0; i < poses.size(); ++i) {
    Sophus2Ros(poses.at(i), parray_msg.poses.at(i));
  }
  pub_parray_->publish(parray_msg);
}

void NodeOdom::PublishCloud(const std_msgs::msg::Header& header) {
  if (pub_points_->get_subscription_count() == 0) return;

  cloud_.header = header;
  cloud_.point_step = 16;
  cloud_.fields = MakePointFields("xyzi");

  RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), odom_.window.MargKf().status().Repr());
  Keyframe2Cloud(odom_.window.MargKf(), cloud_, 50.0);
  pub_points_->publish(cloud_);
}

// void NodeOdom::TfCamCb(const geometry_msgs::msg::Transform& tf_cam_msg) {
//   odom_.camera.baseline_ = -tf_cam_msg.translation.x;
//   ROS_INFO_STREAM(odom_.camera.Repr());
// }

// void NodeOdom::TfImuCb(const geometry_msgs::msg::Transform& tf_imu_msg) {}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  cv::setNumThreads(4);
  rclcpp::spin(std::make_shared<sv::dsol::NodeOdom>());
  rclcpp::shutdown();
}
