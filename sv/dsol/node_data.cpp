#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "sv/dsol/extra.h"
#include "sv/dsol/node_util.h"
#include "sv/ros2/msg_conv.h"
#include "sv/util/dataset.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"

namespace sv::dsol {

using SE3d = Sophus::SE3d;
namespace gm = geometry_msgs::msg;
namespace sm = sensor_msgs::msg;
namespace vm = visualization_msgs::msg;

class NodeData : public rclcpp::Node {
 public:
  NodeData();

  void InitParameters();
  void InitOdom();
  void InitRosIO();
  void InitDataset();

  void PublishOdom(const std_msgs::msg::Header& header, const Sophus::SE3d& Twc);
  void PublishCloud(const std_msgs::msg::Header& header) const;
  void SendTransform(const gm::PoseStamped& pose_msg,
                     const std::string& child_frame);
  void Run();

  bool reverse_{false};
  double freq_{10.0};
  double data_max_depth_{0};
  double cloud_max_depth_{100};
  cv::Range data_range_{0, 0};

  Dataset dataset_;
  MotionModel motion_;
  TumFormatWriter writer_;
  DirectOdometry odom_;

  KeyControl ctrl_;
  std::string frame_{"fixed"};
  std::shared_ptr<tf2_ros::TransformBroadcaster> tfbr_;

  std::shared_ptr<rclcpp::Publisher<rosgraph_msgs::msg::Clock>> clock_pub_;
  std::shared_ptr<rclcpp::Publisher<gm::PoseArray>> pose_array_pub_;
  std::shared_ptr<rclcpp::Publisher<vm::Marker>> align_marker_pub_;
  PosePathPublisher gt_pub_;
  PosePathPublisher kf_pub_;
  PosePathPublisher odom_pub_;

  std::shared_ptr<rclcpp::Publisher<sm::PointCloud2>> points_pub_;
};

NodeData::NodeData() : Node("dsol_data") {
  InitParameters();
  InitRosIO();
  InitDataset();
  InitOdom();

  const int wait_ms = get_parameter_or("wait_ms", 0);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "wait_ms: " << wait_ms);
  ctrl_ = KeyControl(wait_ms);

  const auto save = get_parameter_or<std::string>("save", "");
  writer_ = TumFormatWriter(save);
  if (!writer_.IsDummy()) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("rclcpp"), "Writing results to: " << writer_.filename());
  }

  const auto alpha = get_parameter_or("motion_alpha", 0.5);
  motion_ = MotionModel(alpha);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "motion_alpha: " << motion_.alpha());

  // this is to make camera z pointing forward
  //  const Eigen::Quaterniond q_f_c0(
  //      Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()) *
  //      Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()));
  //  T_f_c0_.setQuaternion(q_f_c0);
  //  ROS_INFO_STREAM("T_f_c0: \n" << T_f_c0_.matrix());

  tfbr_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
}

void NodeData::InitParameters() {
  declare_parameter<int>("tbb");
  declare_parameter<int>("log");
  declare_parameter<int>("vis");
  declare_parameter<double>("freq");
  declare_parameter<std::string>("save");
  declare_parameter<int>("wait_ms");
  declare_parameter<std::string>("data_dir");
  declare_parameter<double>("data_max_depth");
  declare_parameter<double>("cloud_max_depth");
  declare_parameter<double>("motion_alpha");
  declare_parameter<int>("start");
  declare_parameter<int>("end");
  declare_parameter<bool>("reverse");
}

void NodeData::InitRosIO() {
  clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", 1);

  gt_pub_ = PosePathPublisher(*this, "gt", frame_);
  kf_pub_ = PosePathPublisher(*this, "kf", frame_);
  odom_pub_ = PosePathPublisher(*this, "odom", frame_);
  points_pub_ = create_publisher<sm::PointCloud2>("points", 1);
  pose_array_pub_ = create_publisher<gm::PoseArray>("poses", 1);
  align_marker_pub_ = create_publisher<vm::Marker>("align_graph", 1);
}

void NodeData::InitDataset() {
  const auto data_dir = get_parameter_or<std::string>("data_dir", "");
  dataset_ = CreateDataset(data_dir);
  if (!dataset_) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Invalid dataset at: " << data_dir);
    rclcpp::shutdown();
  }
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), dataset_.Repr());

  get_parameter("start", data_range_.start);
  get_parameter("end", data_range_.end);
  get_parameter("reverse", reverse_);

  if (data_range_.end <= 0) {
    data_range_.end += dataset_.size();
  }
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Data range: [%d, %d)", data_range_.start, data_range_.end);
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Reverse: %s", reverse_ ? "true" : "false");

  get_parameter("freq", freq_);
  get_parameter("data_max_depth", data_max_depth_);
  get_parameter("cloud_max_depth", cloud_max_depth_);

  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Freq: " << freq_);
  RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Max depth: " << data_max_depth_);
}

void NodeData::InitOdom() {
  {
    auto cfg = ReadOdomCfg(*this, "odom");
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
}

void NodeData::PublishCloud(const std_msgs::msg::Header& header) const {
  if (points_pub_->get_subscription_count() == 0) return;

  static sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = header;
  cloud.point_step = 16;
  cloud.fields = MakePointFields("xyzi");

  RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), odom_.window.MargKf().status().Repr());
  Keyframe2Cloud(odom_.window.MargKf(), cloud, cloud_max_depth_);
  points_pub_->publish(cloud);
}

void NodeData::SendTransform(const geometry_msgs::msg::PoseStamped& pose_msg,
                             const std::string& child_frame) {
  gm::TransformStamped tf_msg;
  tf_msg.header = pose_msg.header;
  tf_msg.child_frame_id = child_frame;
  Ros2Ros(pose_msg.pose, tf_msg.transform);
  tfbr_->sendTransform(tf_msg);
}

void NodeData::Run() {
  rclcpp::Time time(0,0);
  const auto dt = 1.0 / freq_;
  rclcpp::Duration dtime((int)freq_, 0);

  bool init_tf{false};
  SE3d T_c0_w_gt;
  SE3d dT_pred;

  int start_ind = reverse_ ? data_range_.end - 1 : data_range_.start;
  int end_ind = reverse_ ? data_range_.start - 1 : data_range_.end;
  const int delta = reverse_ ? -1 : 1;

  // Marker
  vm::Marker align_marker;

  for (int ind = start_ind, cnt = 0; ind != end_ind; ind += delta, ++cnt) {
    if (!rclcpp::ok() || !ctrl_.Wait()) break;

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "=== %d ===", ind);
    rosgraph_msgs::msg::Clock clock;
    clock.clock = time;
    clock_pub_->publish(clock);

    // Image
    auto image_l = dataset_.Get(DataType::kImage, ind, 0);
    auto image_r = dataset_.Get(DataType::kImage, ind, 1);

    // Intrin
    if (!odom_.camera.Ok()) {
      const auto intrin = dataset_.Get(DataType::kIntrin, ind);
      const auto camera = Camera::FromMat({image_l.cols, image_l.rows}, intrin);
      odom_.SetCamera(camera);
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), camera);
    }

    // Depth
    auto depth = dataset_.Get(DataType::kDepth, ind, 0);

    if (!depth.empty()) {
      if (data_max_depth_ > 0) {
        cv::threshold(depth, depth, data_max_depth_, 0, cv::THRESH_TOZERO_INV);
      }
    }

    // Pose
    const auto pose_gt = dataset_.Get(DataType::kPose, ind, 0);

    // Record the inverse of the first transform
    if (!init_tf) {
      T_c0_w_gt = SE3dFromMat(pose_gt).inverse();
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "T_c0_w:\n" << T_c0_w_gt.matrix());
      init_tf = true;
    }

    // Then we transform everything into c0 frame
    const auto T_c0_c_gt = T_c0_w_gt * SE3dFromMat(pose_gt);

    // Motion model predict
    if (!motion_.Ok()) {
      motion_.Init(T_c0_c_gt);
    } else {
      dT_pred = motion_.PredictDelta(dt);
    }

    const auto T_pred = odom_.frame.Twc() * dT_pred;

    // Odom
    const auto status = odom_.Estimate(image_l, image_r, dT_pred, depth);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), status.Repr());

    // Motion model correct if tracking is ok and not first frame
    if (status.track.ok && ind != start_ind) {
      motion_.Correct(status.Twc(), dt);
    } else {
      RCLCPP_WARN_STREAM(rclcpp::get_logger("rclcpp"), "Tracking failed (or 1st frame), slow motion model");
      motion_.Scale(0.5);
    }

    // Write to output
    writer_.Write(cnt, status.Twc());

    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), "trans gt:   " << T_c0_c_gt.translation().transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), "trans pred: " << T_pred.translation().transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), "trans odom: " << status.Twc().translation().transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), "trans ba:   "
                     << odom_.window.CurrKf().Twc().translation().transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), "aff_l: " << odom_.frame.state().affine_l.ab.transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), "aff_r: " << odom_.frame.state().affine_r.ab.transpose());

    // publish stuff
    std_msgs::msg::Header header;
    header.frame_id = frame_;
    header.stamp = time;

    gt_pub_.Publish(time, T_c0_c_gt);
    PublishOdom(header, status.Twc());

    if (status.map.remove_kf) {
      PublishCloud(header);
    }

    // Draw align graph
    //    align_marker.header = header;
    //    DrawAlignGraph(status.Twc().translation(),
    //                   odom_.window.GetAllTrans(),
    //                   odom_.aligner.num_tracks(),
    //                   CV_RGB(1.0, 0.0, 0.0),
    //                   0.1,
    //                   align_marker);
    //    align_marker_pub_.publish(align_marker);

    time += dtime;
  }
}

void NodeData::PublishOdom(const std_msgs::msg::Header& header,
                           const Sophus::SE3d& Twc) {
  const auto odom_pose_msg = odom_pub_.Publish(header.stamp, Twc);
  SendTransform(odom_pose_msg, "camera");

  const auto poses = odom_.window.GetAllPoses();
  gm::PoseArray pose_array_msg;
  pose_array_msg.header = header;
  pose_array_msg.poses.resize(poses.size());
  for (size_t i = 0; i < poses.size(); ++i) {
    Sophus2Ros(poses.at(i), pose_array_msg.poses.at(i));
  }
  pose_array_pub_->publish(pose_array_msg);
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  //rclcpp::spin(std::make_shared<sv::dsol::NodeData>());

  const std::shared_ptr<sv::dsol::NodeData>& node =
      std::make_shared<sv::dsol::NodeData>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
    auto spin_executor = [&executor]() {
    executor.spin();
  };
    std::thread execution_thread(spin_executor);
    node->Run();
  execution_thread.join();
  rclcpp::shutdown();
}
