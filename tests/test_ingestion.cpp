#include "prediction/types.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr double kTolerance = 1e-9;

Eigen::Matrix3d rotation_x(double roll) {
  const double c = std::cos(roll);
  const double s = std::sin(roll);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  rotation(1, 1) = c;
  rotation(1, 2) = -s;
  rotation(2, 1) = s;
  rotation(2, 2) = c;
  return rotation;
}

Eigen::Matrix3d rotation_y(double pitch) {
  const double c = std::cos(pitch);
  const double s = std::sin(pitch);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  rotation(0, 0) = c;
  rotation(0, 2) = s;
  rotation(2, 0) = -s;
  rotation(2, 2) = c;
  return rotation;
}

Eigen::Matrix3d rotation_z(double yaw) {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  rotation(0, 0) = c;
  rotation(0, 1) = -s;
  rotation(1, 0) = s;
  rotation(1, 1) = c;
  return rotation;
}

/// Active intrinsic ZYX, equivalent to extrinsic X-then-Y-then-Z.
Eigen::Matrix4d make_se3(double yaw, double pitch, double roll, const Eigen::Vector3d& translation) {
  const Eigen::Matrix3d rotation = rotation_z(yaw) * rotation_y(pitch) * rotation_x(roll);
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation;
  transform.block<3, 1>(0, 3) = translation;
  return transform;
}

prediction::Detection make_detection(double x, double y, double heading, double length, double width,
                                     double score) {
  prediction::Detection detection;
  detection.position = Eigen::Vector2d(x, y);
  detection.size = Eigen::Vector2d(length, width);
  detection.heading = heading;
  detection.score = score;
  return detection;
}

void expect_detection_near(const prediction::Detection& actual, const prediction::Detection& expected) {
  EXPECT_NEAR(actual.position.x(), expected.position.x(), kTolerance);
  EXPECT_NEAR(actual.position.y(), expected.position.y(), kTolerance);
  EXPECT_NEAR(actual.size.x(), expected.size.x(), kTolerance);
  EXPECT_NEAR(actual.size.y(), expected.size.y(), kTolerance);
  EXPECT_NEAR(actual.heading, expected.heading, kTolerance);
  EXPECT_NEAR(actual.score, expected.score, kTolerance);
}

}  // namespace

TEST(Ingestion, AngleNormalizationTest) {
  const std::vector<double> inputs = {
      0.0, prediction::kPi, -prediction::kPi, 3.0 * prediction::kPi, -5.0 * prediction::kPi,
      5.5 * prediction::kPi, -7.25 * prediction::kPi, 0.3, -2.2, 100.0, -100.0};

  for (const double input : inputs) {
    const double normalized = prediction::normalize_angle(input);
    EXPECT_GE(normalized, -prediction::kPi);
    EXPECT_LE(normalized, prediction::kPi);
    EXPECT_TRUE(std::isfinite(normalized));
    EXPECT_NEAR(std::cos(normalized), std::cos(input), kTolerance);
    EXPECT_NEAR(std::sin(normalized), std::sin(input), kTolerance);
  }

  const std::vector<double> boundaries = {prediction::kPi, -prediction::kPi, 3.0 * prediction::kPi,
                                           -5.0 * prediction::kPi};
  for (const double boundary : boundaries) {
    const double normalized = prediction::normalize_angle(boundary);
    EXPECT_LT(std::abs(std::abs(normalized) - prediction::kPi), kTolerance);
  }

  EXPECT_NEAR(prediction::normalize_angle(5.5 * prediction::kPi), -0.5 * prediction::kPi, kTolerance);
  EXPECT_NEAR(prediction::normalize_angle(-7.25 * prediction::kPi), 0.75 * prediction::kPi, kTolerance);
  EXPECT_NEAR(prediction::normalize_angle(0.0), 0.0, kTolerance);
  EXPECT_NEAR(prediction::normalize_angle(prediction::kTwoPi), 0.0, kTolerance);
}

TEST(Ingestion, SE2Transform_RoundTripAndInvariance) {
  const prediction::Detection world_target =
      make_detection(30.0, -10.0, 0.25 * prediction::kPi, 4.8, 1.9, 0.93);

  const std::vector<prediction::Pose2D> ego_poses = {
      prediction::Pose2D{0.0, 0.0, 0.0},
      prediction::Pose2D{10.0, 5.0, 0.5 * prediction::kPi},
      prediction::Pose2D{-15.0, 20.0, -0.75 * prediction::kPi},
  };

  for (const prediction::Pose2D& ego_pose : ego_poses) {
    const prediction::Detection in_ego = prediction::transform_world_to_ego(world_target, ego_pose);
    const prediction::Detection recovered = prediction::transform_ego_to_world(in_ego, ego_pose);
    expect_detection_near(recovered, world_target);

    const prediction::Detection back_to_ego = prediction::transform_world_to_ego(recovered, ego_pose);
    expect_detection_near(back_to_ego, in_ego);
  }
}

TEST(Ingestion, SE3Transform_EquivalenceWithSE2) {
  const prediction::Pose2D ego_pose{3.5, -2.25, 0.4};
  const prediction::Detection det_ego = make_detection(8.0, -3.0, 1.2, 4.2, 1.7, 0.61);
  const Eigen::Matrix4d ego_to_world =
      make_se3(ego_pose.yaw, 0.0, 0.0, Eigen::Vector3d(ego_pose.x, ego_pose.y, 0.0));

  const prediction::Detection from_se2 = prediction::transform_ego_to_world(det_ego, ego_pose);
  const prediction::Detection from_se3 = prediction::transform_ego_to_world(det_ego, ego_to_world);
  expect_detection_near(from_se3, from_se2);
}

TEST(Ingestion, SE3Transform_PitchRollProjection) {
  const double yaw = 30.0 * prediction::kPi / 180.0;
  const double pitch = 5.0 * prediction::kPi / 180.0;
  const double roll = 10.0 * prediction::kPi / 180.0;
  const Eigen::Vector3d translation(10.0, 20.0, 2.5);
  const Eigen::Matrix4d ego_to_world = make_se3(yaw, pitch, roll, translation);
  const Eigen::Matrix3d rotation = ego_to_world.block<3, 3>(0, 0);

  const prediction::Detection level_heading = make_detection(15.0, 2.0, 0.0, 5.1, 2.0, 0.77);
  const Eigen::Vector3d expected_point =
      rotation * Eigen::Vector3d(level_heading.position.x(), level_heading.position.y(), 0.0) +
      translation;
  const Eigen::Vector3d expected_forward = rotation * Eigen::Vector3d(1.0, 0.0, 0.0);

  const prediction::Detection projected =
      prediction::transform_ego_to_world(level_heading, ego_to_world);

  EXPECT_NEAR(projected.position.x(), expected_point.x(), kTolerance);
  EXPECT_NEAR(projected.position.y(), expected_point.y(), kTolerance);
  EXPECT_NEAR(projected.heading,
              prediction::normalize_angle(std::atan2(expected_forward.y(), expected_forward.x())),
              kTolerance);
  EXPECT_NEAR(projected.size.x(), level_heading.size.x(), kTolerance);
  EXPECT_NEAR(projected.size.y(), level_heading.size.y(), kTolerance);
  EXPECT_NEAR(projected.score, level_heading.score, kTolerance);
  EXPECT_TRUE(std::isfinite(projected.position.x()));
  EXPECT_TRUE(std::isfinite(projected.position.y()));
  EXPECT_TRUE(std::isfinite(projected.heading));
  EXPECT_GE(projected.heading, -prediction::kPi);
  EXPECT_LE(projected.heading, prediction::kPi);

  // A zero ego heading only exercises the transform's own yaw. A nonzero local
  // heading checks that the forward vector is rotated before projection.
  const double ego_heading = 0.35;
  const prediction::Detection angled = make_detection(15.0, 2.0, ego_heading, 5.1, 2.0, 0.77);
  const Eigen::Vector3d angled_forward =
      rotation * Eigen::Vector3d(std::cos(ego_heading), std::sin(ego_heading), 0.0);
  const prediction::Detection angled_projected =
      prediction::transform_ego_to_world(angled, ego_to_world);
  EXPECT_NEAR(angled_projected.heading,
              prediction::normalize_angle(std::atan2(angled_forward.y(), angled_forward.x())),
              kTolerance);
  EXPECT_GT(std::abs(angled_projected.heading - projected.heading), 1e-3);
}
