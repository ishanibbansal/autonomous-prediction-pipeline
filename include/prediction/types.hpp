#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace prediction {

/// Portable pi. C++17 has neither std::numbers::pi nor a standard M_PI.
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 2.0 * kPi;

/// Planar ego pose. yaw is radians and is expected in [-pi, pi].
struct Pose2D {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

/// One object observation in either the ego or world BEV frame.
/// position is [x, y]^T in meters. size is [length, width]^T in meters.
/// heading is yaw in radians, wrapped to [-pi, pi]. score is confidence in [0, 1].
struct Detection {
  Eigen::Vector2d position{Eigen::Vector2d::Zero()};
  Eigen::Vector2d size{Eigen::Vector2d::Zero()};
  double heading{0.0};
  double score{0.0};
};

/// One synchronized detection snapshot.
/// ego_pose_se3 is the upstream ego-to-world rigid transform. Its planar
/// projection is not required to match ego_pose_2d; callers choose which one
/// to apply.
struct DetectionFrame {
  std::uint64_t frame_id{0};
  double timestamp{0.0};
  Pose2D ego_pose_2d{};
  Eigen::Matrix4d ego_pose_se3{Eigen::Matrix4d::Identity()};
  std::vector<Detection> detections{};
};

enum class TrackStatus { Tentative, Confirmed, Lost };

/// Downstream track contract.
/// state is [x, y, v_x, v_y, psi, omega]^T in meters, m/s, radians, and rad/s.
struct TrackState {
  std::uint64_t track_id{0};
  TrackStatus status{TrackStatus::Tentative};
  Eigen::Vector2d size{Eigen::Vector2d::Zero()};
  Eigen::Matrix<double, 6, 1> state{Eigen::Matrix<double, 6, 1>::Zero()};
  Eigen::Matrix<double, 6, 6> covariance{Eigen::Matrix<double, 6, 6>::Identity()};
};

struct PredictedWaypoint {
  double timestamp{0.0};
  Eigen::Vector2d position{Eigen::Vector2d::Zero()};
  double heading{0.0};
  Eigen::Matrix2d position_covariance{Eigen::Matrix2d::Identity()};
};

/// One future sample of a kinematic trajectory mode.
/// position is meters, velocity is m/s, heading is radians in the range returned
/// by normalize_angle, timestamp is seconds, and position_covariance is m^2.
struct TrajectoryPoint {
  Eigen::Vector2d position{Eigen::Vector2d::Zero()};
  Eigen::Vector2d velocity{Eigen::Vector2d::Zero()};
  double heading{0.0};
  double timestamp{0.0};
  Eigen::Matrix2d position_covariance{Eigen::Matrix2d::Zero()};
};

/// One geometric motion hypothesis and its heuristic mixture weight.
/// probability is not a calibrated behavioral probability.
struct TrajectoryMode {
  double probability{0.0};
  std::vector<TrajectoryPoint> waypoints{};
};

/// Downstream trajectory contract.
///
/// The fields above `modes` are the Milestone 1 contract and keep their order.
/// `waypoints` remains the legacy single-trajectory view. A predictor-produced
/// result also fills `modes`, which is the authoritative multimodal trajectory.
/// That predictor copies the straight mode into `waypoints` without velocity, so
/// the straight samples are stored twice. A default-constructed value has neither
/// legacy waypoints nor modes; emptiness is not a predictor error code.
struct PredictedTrajectory {
  std::uint64_t track_id{0};
  TrackState current_state{};
  std::vector<PredictedWaypoint> waypoints{};
  std::vector<TrajectoryMode> modes{};
};

/// Wraps a finite angle into [-pi, pi]. +/- pi are equivalent and either
/// endpoint may be returned for odd multiples of pi.
inline double normalize_angle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

/// SE(2) ego-to-world. size and score are copied unchanged.
inline Detection transform_ego_to_world(const Detection& det_ego, const Pose2D& ego_to_world) {
  const double theta = ego_to_world.yaw;
  const double c = std::cos(theta);
  const double s = std::sin(theta);

  Detection world = det_ego;
  world.position.x() = c * det_ego.position.x() - s * det_ego.position.y() + ego_to_world.x;
  world.position.y() = s * det_ego.position.x() + c * det_ego.position.y() + ego_to_world.y;
  world.heading = normalize_angle(det_ego.heading + theta);
  return world;
}

/// Inverse of transform_ego_to_world(..., Pose2D). ego_to_world is still the
/// ego-to-world pose, not the world-to-ego pose.
inline Detection transform_world_to_ego(const Detection& det_world, const Pose2D& ego_to_world) {
  const double theta = ego_to_world.yaw;
  const double c = std::cos(-theta);
  const double s = std::sin(-theta);
  const double delta_x = det_world.position.x() - ego_to_world.x;
  const double delta_y = det_world.position.y() - ego_to_world.y;

  Detection ego = det_world;
  ego.position.x() = c * delta_x - s * delta_y;
  ego.position.y() = s * delta_x + c * delta_y;
  ego.heading = normalize_angle(det_world.heading - theta);
  return ego;
}

/// Projects an ego-frame detection through a full SE(3) ego-to-world transform.
/// The detection is treated as lying on the ego z = 0 plane. World z is dropped
/// so the result is a BEV observation. Heading is the yaw of the transformed
/// local forward vector, not an Euler extraction of the transform alone.
/// size and score are copied unchanged; pitched boxes are not reprojected.
inline Detection transform_ego_to_world(const Detection& det_ego,
                                        const Eigen::Matrix4d& ego_to_world_se3) {
  Eigen::Vector4d point_ego;
  point_ego << det_ego.position.x(), det_ego.position.y(), 0.0, 1.0;
  const Eigen::Vector4d point_world = ego_to_world_se3 * point_ego;

  Eigen::Vector4d forward_ego;
  forward_ego << std::cos(det_ego.heading), std::sin(det_ego.heading), 0.0, 0.0;
  const Eigen::Vector4d forward_world = ego_to_world_se3 * forward_ego;

  Detection world = det_ego;
  world.position.x() = point_world(0);
  world.position.y() = point_world(1);
  world.heading = normalize_angle(std::atan2(forward_world(1), forward_world(0)));
  return world;
}

}  // namespace prediction
