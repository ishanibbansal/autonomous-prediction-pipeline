#include "prediction/predictor.hpp"
#include "prediction/tracker.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace prediction {
namespace test_support {
const KinematicTrajectoryPredictor& predictor_linkage_anchor();
}  // namespace test_support
}  // namespace prediction

namespace {

constexpr double kPositionTolerance = 1e-8;
constexpr double kLooseTolerance = 1e-6;
constexpr double kCovarianceTolerance = 1e-9;

prediction::TrackState make_track(std::uint64_t track_id, double x, double y, double vx, double vy,
                                  double heading, double omega,
                                  prediction::TrackStatus status = prediction::TrackStatus::Confirmed) {
  prediction::TrackState track;
  track.track_id = track_id;
  track.status = status;
  track.size = Eigen::Vector2d(4.5, 1.9);
  track.state << x, y, vx, vy, heading, omega;
  track.covariance.setZero();
  track.covariance(0, 0) = 0.25;
  track.covariance(1, 1) = 0.25;
  track.covariance(2, 2) = 1.0;
  track.covariance(3, 3) = 1.0;
  track.covariance(4, 4) = 0.01;
  track.covariance(5, 5) = 0.01;
  return track;
}

void expect_config_rejected(const prediction::PredictorConfig& config, const char* field) {
  try {
    prediction::KinematicTrajectoryPredictor predictor(config);
    FAIL() << "expected std::invalid_argument naming " << field;
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(field), std::string::npos) << error.what();
  }
}

void expect_track_rejected(const prediction::TrackState& track, double timestamp, const char* field) {
  const prediction::KinematicTrajectoryPredictor predictor;
  const prediction::TrackState before = track;
  try {
    predictor.predict(track, timestamp);
    FAIL() << "expected std::invalid_argument naming " << field;
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(field), std::string::npos) << error.what();
  }
  EXPECT_EQ(track.track_id, before.track_id);
  EXPECT_EQ(track.status, before.status);
  EXPECT_DOUBLE_EQ(track.state(4), before.state(4));
  EXPECT_DOUBLE_EQ(track.covariance(0, 1), before.covariance(0, 1));
  EXPECT_DOUBLE_EQ(track.covariance(1, 0), before.covariance(1, 0));
  EXPECT_DOUBLE_EQ(track.size.x(), before.size.x());
}

Eigen::Vector2d rotate(const Eigen::Vector2d& velocity, double turn) {
  return Eigen::Vector2d(std::cos(turn) * velocity.x() - std::sin(turn) * velocity.y(),
                         std::sin(turn) * velocity.x() + std::cos(turn) * velocity.y());
}

/// Independent circular-arc integral, not the predictor's sinc form.
Eigen::Vector2d analytical_displacement(const Eigen::Vector2d& velocity, double rate, double offset) {
  if (rate == 0.0) {
    return offset * velocity;
  }
  const double turn = rate * offset;
  const double sine = std::sin(turn);
  const double cosine = std::cos(turn);
  Eigen::Matrix2d integral;
  integral << sine, -(1.0 - cosine), 1.0 - cosine, sine;
  return (integral * velocity) / rate;
}

void expect_heading_range(double heading) {
  EXPECT_GE(heading, -prediction::kPi);
  EXPECT_LE(heading, prediction::kPi);
  EXPECT_TRUE(std::isfinite(heading));
}

prediction::Detection make_detection(double x, double y, double heading) {
  prediction::Detection detection;
  detection.position = Eigen::Vector2d(x, y);
  detection.size = Eigen::Vector2d(4.5, 1.9);
  detection.heading = heading;
  detection.score = 0.9;
  return detection;
}

}  // namespace

TEST(KinematicTrajectoryPredictor, DefaultConfigIsAcceptedAndProbabilitiesNormalize) {
  const prediction::KinematicTrajectoryPredictor predictor;
  const prediction::PredictorConfig& config = predictor.config();
  EXPECT_DOUBLE_EQ(config.horizon_seconds, 3.0);
  EXPECT_DOUBLE_EQ(config.step_size_seconds, 0.1);
  EXPECT_DOUBLE_EQ(config.branch_turn_rate, 0.15);
  EXPECT_DOUBLE_EQ(config.pos_diffusion_rate, 0.25);
  EXPECT_NEAR(config.prob_straight + config.prob_left + config.prob_right, 1.0, 1e-15);

  prediction::PredictorConfig slightly_off;
  slightly_off.prob_straight = 0.60 + 5e-7;
  slightly_off.prob_left = 0.20;
  slightly_off.prob_right = 0.20 - 5e-7;
  const prediction::KinematicTrajectoryPredictor normalized(slightly_off);
  const prediction::PredictorConfig& stored = normalized.config();
  EXPECT_NEAR(stored.prob_straight + stored.prob_left + stored.prob_right, 1.0, 1e-15);
  EXPECT_NEAR(stored.prob_straight / stored.prob_left,
              slightly_off.prob_straight / slightly_off.prob_left, 1e-12);
}

TEST(KinematicTrajectoryPredictor, ConfigValidationNamesTheRejectedField) {
  prediction::PredictorConfig config;

  config.horizon_seconds = 0.0;
  expect_config_rejected(config, "PredictorConfig::horizon_seconds");
  config.horizon_seconds = -1.0;
  expect_config_rejected(config, "PredictorConfig::horizon_seconds");
  config.horizon_seconds = std::numeric_limits<double>::quiet_NaN();
  expect_config_rejected(config, "PredictorConfig::horizon_seconds");
  config.horizon_seconds = std::numeric_limits<double>::infinity();
  expect_config_rejected(config, "PredictorConfig::horizon_seconds");

  config = {};
  config.step_size_seconds = 0.0;
  expect_config_rejected(config, "PredictorConfig::step_size_seconds");
  config.step_size_seconds = std::numeric_limits<double>::quiet_NaN();
  expect_config_rejected(config, "PredictorConfig::step_size_seconds");

  config = {};
  config.horizon_seconds = 0.05;
  config.step_size_seconds = 0.1;
  expect_config_rejected(config, "PredictorConfig::horizon_seconds");

  config = {};
  config.branch_turn_rate = 0.0;
  expect_config_rejected(config, "PredictorConfig::branch_turn_rate");
  config.branch_turn_rate = -0.2;
  expect_config_rejected(config, "PredictorConfig::branch_turn_rate");
  config.branch_turn_rate = std::numeric_limits<double>::infinity();
  expect_config_rejected(config, "PredictorConfig::branch_turn_rate");

  config = {};
  config.prob_left = 0.0;
  expect_config_rejected(config, "PredictorConfig::prob_left");
  config = {};
  config.prob_right = -0.1;
  expect_config_rejected(config, "PredictorConfig::prob_right");
  config = {};
  config.prob_straight = 1.1;
  expect_config_rejected(config, "PredictorConfig::prob_straight");
  config = {};
  config.prob_straight = 0.5;
  config.prob_left = 0.5;
  config.prob_right = 0.5;
  expect_config_rejected(config, "PredictorConfig probabilities");
  config.prob_straight = std::numeric_limits<double>::quiet_NaN();
  expect_config_rejected(config, "PredictorConfig::prob_straight");

  config = {};
  config.pos_diffusion_rate = -1e-9;
  expect_config_rejected(config, "PredictorConfig::pos_diffusion_rate");
  config.pos_diffusion_rate = std::numeric_limits<double>::quiet_NaN();
  expect_config_rejected(config, "PredictorConfig::pos_diffusion_rate");

  config = {};
  config.horizon_seconds = 1e300;
  config.step_size_seconds = 1e-300;
  expect_config_rejected(config, "PredictorConfig::horizon_seconds");

  config = {};
  config.horizon_seconds = 2.0;
  config.step_size_seconds = 0.0001;
  expect_config_rejected(config, "kMaxPredictionSteps");

  config = {};
  config.horizon_seconds = 1.0;
  config.step_size_seconds = 1e-4;
  EXPECT_NO_THROW(prediction::KinematicTrajectoryPredictor{config});
}

TEST(KinematicTrajectoryPredictor, StraightModePreservesWorldVelocity) {
  const prediction::KinematicTrajectoryPredictor predictor;
  const prediction::TrackState track = make_track(7, 0.0, 0.0, 10.0, 0.0, 0.4, 0.8);
  const prediction::PredictedTrajectory forecast = predictor.predict(track, 0.0);

  ASSERT_EQ(forecast.modes.size(), 3U);
  ASSERT_EQ(forecast.modes[0].waypoints.size(), 30U);
  ASSERT_EQ(forecast.waypoints.size(), 30U);
  EXPECT_EQ(forecast.track_id, 7U);
  EXPECT_EQ(forecast.current_state.status, prediction::TrackStatus::Confirmed);
  EXPECT_DOUBLE_EQ(forecast.current_state.state(4), 0.4);
  EXPECT_DOUBLE_EQ(forecast.current_state.state(5), 0.8);

  const prediction::TrajectoryPoint& last = forecast.modes[0].waypoints.back();
  EXPECT_NEAR(last.position.x(), 30.0, kLooseTolerance);
  EXPECT_NEAR(last.position.y(), 0.0, kPositionTolerance);
  EXPECT_DOUBLE_EQ(last.velocity.x(), 10.0);
  EXPECT_DOUBLE_EQ(last.velocity.y(), 0.0);
  EXPECT_NEAR(last.heading, 0.4, 1e-15);
  EXPECT_NEAR(last.timestamp, 3.0, kLooseTolerance);
  EXPECT_NEAR(forecast.waypoints.back().position.x(), last.position.x(), 0.0);
  EXPECT_NEAR(forecast.waypoints.front().timestamp, 0.1, kLooseTolerance);
}

TEST(KinematicTrajectoryPredictor, HeadingDoesNotRedirectWorldVelocity) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 1.0;
  config.step_size_seconds = 0.25;
  const prediction::KinematicTrajectoryPredictor predictor(config);

  const prediction::TrackState sideways = make_track(1, 2.0, -3.0, 0.0, 4.0, 0.0, 0.0);
  const prediction::TrackState reverse = make_track(1, 2.0, -3.0, -6.0, 0.0, 0.0, 0.3);
  const auto side = predictor.predict(sideways, 1.5);
  const auto back = predictor.predict(reverse, 1.5);

  ASSERT_EQ(side.modes[0].waypoints.size(), 4U);
  const prediction::TrajectoryPoint& side_last = side.modes[0].waypoints.back();
  EXPECT_NEAR(side_last.position.x(), 2.0, kPositionTolerance);
  EXPECT_NEAR(side_last.position.y(), 1.0, kLooseTolerance);
  EXPECT_NEAR(side_last.velocity.y(), 4.0, kPositionTolerance);
  EXPECT_NEAR(side_last.heading, 0.0, kPositionTolerance);

  const prediction::TrajectoryPoint& reverse_last = back.modes[0].waypoints.back();
  EXPECT_NEAR(reverse_last.position.x(), -4.0, kLooseTolerance);
  EXPECT_NEAR(reverse_last.position.y(), -3.0, kPositionTolerance);
  EXPECT_NEAR(reverse_last.velocity.x(), -6.0, kPositionTolerance);

  prediction::TrackState other_heading = sideways;
  other_heading.state(4) = 1.2;
  const auto turned_heading = predictor.predict(other_heading, 1.5);
  for (std::size_t step = 0; step < side.modes[0].waypoints.size(); ++step) {
    EXPECT_NEAR(turned_heading.modes[0].waypoints[step].position.x(),
                side.modes[0].waypoints[step].position.x(), kPositionTolerance);
    EXPECT_NEAR(turned_heading.modes[0].waypoints[step].position.y(),
                side.modes[0].waypoints[step].position.y(), kPositionTolerance);
  }
}

TEST(KinematicTrajectoryPredictor, CircularArcMatchesIndependentIntegral) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 3.0;
  config.step_size_seconds = 0.5;
  config.branch_turn_rate = 0.15;
  const prediction::KinematicTrajectoryPredictor predictor(config);

  const double rate = prediction::kPi / 6.0;
  const Eigen::Vector2d velocity(10.0, 0.0);
  const prediction::TrackState track = make_track(3, 1.0, -2.0, velocity.x(), velocity.y(), 0.2, rate);
  const prediction::PredictedTrajectory forecast = predictor.predict(track, 0.0);
  ASSERT_EQ(forecast.modes[1].waypoints.size(), 6U);

  for (std::size_t step = 0; step < forecast.modes[1].waypoints.size(); ++step) {
    const double offset = static_cast<double>(step + 1U) * config.step_size_seconds;
    const Eigen::Vector2d expected =
        Eigen::Vector2d(1.0, -2.0) + analytical_displacement(velocity, rate, offset);
    const prediction::TrajectoryPoint& point = forecast.modes[1].waypoints[step];
    EXPECT_NEAR(point.position.x(), expected.x(), kPositionTolerance) << "step " << step;
    EXPECT_NEAR(point.position.y(), expected.y(), kPositionTolerance) << "step " << step;
    const Eigen::Vector2d expected_velocity = rotate(velocity, rate * offset);
    EXPECT_NEAR(point.velocity.x(), expected_velocity.x(), kPositionTolerance);
    EXPECT_NEAR(point.velocity.y(), expected_velocity.y(), kPositionTolerance);
    EXPECT_NEAR(point.velocity.norm(), velocity.norm(), kPositionTolerance);
    expect_heading_range(point.heading);
  }

  const prediction::TrajectoryPoint& quarter = forecast.modes[1].waypoints.back();
  const double radius = 10.0 / rate;
  EXPECT_NEAR(quarter.position.x(), 1.0 + radius, kLooseTolerance);
  EXPECT_NEAR(quarter.position.y(), -2.0 + radius, kLooseTolerance);

  const prediction::TrackState clockwise =
      make_track(4, 1.0, -2.0, velocity.x(), velocity.y(), 0.2, -rate);
  const prediction::PredictedTrajectory right = predictor.predict(clockwise, 0.0);
  const prediction::TrajectoryPoint& right_last = right.modes[2].waypoints.back();
  EXPECT_NEAR(right_last.position.x(), 1.0 + radius, kLooseTolerance);
  EXPECT_NEAR(right_last.position.y(), -2.0 - radius, kLooseTolerance);
  EXPECT_NEAR(right_last.velocity.y(), -10.0, kLooseTolerance);
}

TEST(KinematicTrajectoryPredictor, GeneralVelocityFollowsBothTurnSigns) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 1.0;
  config.step_size_seconds = 0.25;
  config.branch_turn_rate = 0.1;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  const Eigen::Vector2d velocity(3.0, 4.0);
  const prediction::TrackState track =
      make_track(8, 10.0, -2.0, velocity.x(), velocity.y(), -0.7, 0.4);
  const prediction::PredictedTrajectory forecast = predictor.predict(track, -5.0);

  for (std::size_t step = 0; step < forecast.modes[1].waypoints.size(); ++step) {
    const double offset = static_cast<double>(step + 1U) * 0.25;
    const Eigen::Vector2d expected_left =
        Eigen::Vector2d(10.0, -2.0) + analytical_displacement(velocity, 0.4, offset);
    const Eigen::Vector2d expected_right =
        Eigen::Vector2d(10.0, -2.0) + analytical_displacement(velocity, -0.1, offset);
    EXPECT_NEAR(forecast.modes[1].waypoints[step].position.x(), expected_left.x(), kPositionTolerance);
    EXPECT_NEAR(forecast.modes[1].waypoints[step].position.y(), expected_left.y(), kPositionTolerance);
    EXPECT_NEAR(forecast.modes[2].waypoints[step].position.x(), expected_right.x(), kPositionTolerance);
    EXPECT_NEAR(forecast.modes[2].waypoints[step].position.y(), expected_right.y(), kPositionTolerance);
    EXPECT_NEAR(forecast.modes[1].waypoints[step].timestamp, -5.0 + offset, kLooseTolerance);
  }
}

TEST(KinematicTrajectoryPredictor, TinyBranchRatesStayFiniteAndNearlyStraight) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 2.0;
  config.step_size_seconds = 0.5;
  config.branch_turn_rate = 1e-8;
  const prediction::KinematicTrajectoryPredictor predictor(config);

  for (const double omega : {0.0, 1e-12, -1e-12}) {
    const prediction::TrackState track = make_track(2, 0.0, 0.0, 10.0, -2.0, 0.3, omega);
    const prediction::PredictedTrajectory forecast = predictor.predict(track, 0.0);
    for (const prediction::TrajectoryMode& mode : forecast.modes) {
      for (const prediction::TrajectoryPoint& point : mode.waypoints) {
        EXPECT_TRUE(point.position.allFinite());
        EXPECT_TRUE(point.velocity.allFinite());
        EXPECT_TRUE(std::isfinite(point.heading));
        EXPECT_TRUE(point.position_covariance.allFinite());
        const double offset = point.timestamp;
        EXPECT_NEAR(point.position.x(), 10.0 * offset, 1e-6);
        EXPECT_NEAR(point.position.y(), -2.0 * offset, 1e-6);
      }
    }
  }
}

TEST(KinematicTrajectoryPredictor, SeriesAndDirectSincBranchesMatchTheClosedForm) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 1.5e-3;
  config.step_size_seconds = 5.0e-4;
  config.branch_turn_rate = 0.2;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  const Eigen::Vector2d velocity(8.0, -3.0);
  const prediction::TrackState track = make_track(1, 0.0, 1.0, velocity.x(), velocity.y(), 0.0, 0.0);
  const prediction::PredictedTrajectory forecast = predictor.predict(track, 0.0);
  ASSERT_EQ(forecast.modes[1].waypoints.size(), 3U);

  const double half_turn_at_boundary =
      0.5 * config.branch_turn_rate * (2.0 * config.step_size_seconds);
  EXPECT_NEAR(half_turn_at_boundary, prediction::KinematicTrajectoryPredictor::kSincSeriesThreshold,
              1e-15);

  for (std::size_t step = 0; step < 3U; ++step) {
    const double offset = static_cast<double>(step + 1U) * config.step_size_seconds;
    const Eigen::Vector2d expected =
        Eigen::Vector2d(0.0, 1.0) + analytical_displacement(velocity, config.branch_turn_rate, offset);
    EXPECT_NEAR(forecast.modes[1].waypoints[step].position.x(), expected.x(), 1e-9);
    EXPECT_NEAR(forecast.modes[1].waypoints[step].position.y(), expected.y(), 1e-9);
  }
}

TEST(KinematicTrajectoryPredictor, ZeroTurnModesAreMirrorImages) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 2.0;
  config.step_size_seconds = 0.2;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  const prediction::PredictedTrajectory forecast =
      predictor.predict(make_track(1, 5.0, 1.0, 10.0, 0.0, 0.0, 0.0), 0.0);

  for (std::size_t step = 0; step < forecast.modes[0].waypoints.size(); ++step) {
    const prediction::TrajectoryPoint& left = forecast.modes[1].waypoints[step];
    const prediction::TrajectoryPoint& right = forecast.modes[2].waypoints[step];
    const prediction::TrajectoryPoint& straight = forecast.modes[0].waypoints[step];
    EXPECT_NEAR(left.position.x(), right.position.x(), kPositionTolerance);
    EXPECT_NEAR(left.position.y(), -right.position.y() + 2.0, kPositionTolerance);
    EXPECT_NEAR(left.velocity.x(), right.velocity.x(), kPositionTolerance);
    EXPECT_NEAR(left.velocity.y(), -right.velocity.y(), kPositionTolerance);
    EXPECT_NEAR(left.heading, -right.heading, kPositionTolerance);
    EXPECT_GT(std::abs(left.position.y() - straight.position.y()), 0.0);
  }
  EXPECT_GT(std::abs(forecast.modes[1].waypoints.back().position.y() -
                      forecast.modes[0].waypoints.back().position.y()),
            0.1);
}

TEST(KinematicTrajectoryPredictor, BranchRatesUseMaxAndMinNotAnOffset) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 1.0;
  config.step_size_seconds = 0.5;
  config.branch_turn_rate = 0.15;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  const Eigen::Vector2d velocity(4.0, 0.0);

  const auto above = predictor.predict(make_track(1, 0.0, 0.0, velocity.x(), 0.0, 0.0, 0.5), 0.0);
  const auto below = predictor.predict(make_track(1, 0.0, 0.0, velocity.x(), 0.0, 0.0, -0.5), 0.0);
  const auto at_branch =
      predictor.predict(make_track(1, 0.0, 0.0, velocity.x(), 0.0, 0.0, 0.15), 0.0);

  const Eigen::Vector2d expected_fast =
      analytical_displacement(velocity, 0.5, config.horizon_seconds);
  const Eigen::Vector2d expected_branch =
      analytical_displacement(velocity, 0.15, config.horizon_seconds);
  EXPECT_NEAR(above.modes[1].waypoints.back().position.y(), expected_fast.y(), kPositionTolerance);
  EXPECT_NEAR(above.modes[2].waypoints.back().position.y(), -expected_branch.y(), kPositionTolerance);
  EXPECT_NEAR(below.modes[2].waypoints.back().position.y(), -expected_fast.y(), kPositionTolerance);
  EXPECT_NEAR(below.modes[1].waypoints.back().position.y(), expected_branch.y(), kPositionTolerance);
  EXPECT_NEAR(at_branch.modes[1].waypoints.back().position.y(), expected_branch.y(),
              kPositionTolerance);
  EXPECT_NEAR(above.modes[0].waypoints.back().position.x(), below.modes[0].waypoints.back().position.x(),
              kPositionTolerance);
  EXPECT_NEAR(above.modes[0].waypoints.back().velocity.x(), velocity.x(), kPositionTolerance);
}

TEST(KinematicTrajectoryPredictor, HeadingWrapsWithoutForcingAnEndpoint) {
  const prediction::KinematicTrajectoryPredictor predictor;
  const double initial = 0.95 * prediction::kPi;
  const prediction::PredictedTrajectory forecast =
      predictor.predict(make_track(1, 0.0, 0.0, 1.0, 0.0, initial, 0.0), 0.0);
  EXPECT_NEAR(forecast.modes[0].waypoints.front().heading, prediction::normalize_angle(initial),
              1e-12);
  expect_heading_range(forecast.modes[1].waypoints.back().heading);
  EXPECT_LT(forecast.modes[1].waypoints.back().heading, 0.0);

  prediction::TrackState odd_pi = make_track(1, 0.0, 0.0, 1.0, 0.0, 3.0 * prediction::kPi, 0.0);
  const prediction::PredictedTrajectory wrapped = predictor.predict(odd_pi, 0.0);
  EXPECT_LT(std::abs(std::abs(wrapped.modes[0].waypoints.front().heading) - prediction::kPi), 1e-9);
  EXPECT_DOUBLE_EQ(wrapped.current_state.state(4), 3.0 * prediction::kPi);
}

TEST(KinematicTrajectoryPredictor, ExactStandstillDoesNotRotateAndSlowMotionDoes) {
  const prediction::KinematicTrajectoryPredictor predictor;
  const prediction::TrackState stopped = make_track(9, 4.0, 5.0, 0.0, 0.0, 1.2, 0.4);
  const prediction::PredictedTrajectory still = predictor.predict(stopped, 2.0);
  for (const prediction::TrajectoryMode& mode : still.modes) {
    for (const prediction::TrajectoryPoint& point : mode.waypoints) {
      EXPECT_DOUBLE_EQ(point.position.x(), 4.0);
      EXPECT_DOUBLE_EQ(point.position.y(), 5.0);
      EXPECT_DOUBLE_EQ(point.velocity.x(), 0.0);
      EXPECT_DOUBLE_EQ(point.velocity.y(), 0.0);
      EXPECT_NEAR(point.heading, prediction::normalize_angle(stopped.state(4)), 1e-15);
    }
  }

  const prediction::TrackState creeping = make_track(9, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0);
  const prediction::PredictedTrajectory slow = predictor.predict(creeping, 0.0);
  EXPECT_LT(slow.modes[0].waypoints.back().position.y(), -0.1);
  EXPECT_GT(std::abs(slow.modes[1].waypoints.back().position.x()), 1e-3);
}

TEST(KinematicTrajectoryPredictor, CovarianceFollowsTheSharedHeuristic) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 1.0;
  config.step_size_seconds = 0.25;
  config.pos_diffusion_rate = 0.3;
  const prediction::KinematicTrajectoryPredictor predictor(config);

  prediction::TrackState track = make_track(1, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0);
  track.covariance.setZero();
  track.covariance(0, 0) = 2.0;
  track.covariance(0, 1) = 0.3;
  track.covariance(1, 0) = 0.3;
  track.covariance(1, 1) = 1.5;
  track.covariance(2, 2) = 0.4;
  track.covariance(2, 3) = -0.1;
  track.covariance(3, 2) = -0.1;
  track.covariance(3, 3) = 0.2;
  track.covariance(4, 4) = 0.05;
  track.covariance(5, 5) = 0.08;

  const prediction::PredictedTrajectory forecast = predictor.predict(track, 0.0);
  Eigen::Matrix2d position_block;
  position_block << 2.0, 0.3, 0.3, 1.5;
  Eigen::Matrix2d velocity_block;
  velocity_block << 0.4, -0.1, -0.1, 0.2;
  for (std::size_t step = 0; step < forecast.waypoints.size(); ++step) {
    const double offset = static_cast<double>(step + 1U) * config.step_size_seconds;
    const Eigen::Matrix2d expected =
        position_block + offset * offset * velocity_block +
        config.pos_diffusion_rate * offset * Eigen::Matrix2d::Identity();
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 2; ++col) {
        EXPECT_NEAR(forecast.modes[0].waypoints[step].position_covariance(row, col),
                    expected(row, col), kCovarianceTolerance);
        EXPECT_NEAR(forecast.modes[1].waypoints[step].position_covariance(row, col),
                    expected(row, col), kCovarianceTolerance);
        EXPECT_NEAR(forecast.modes[2].waypoints[step].position_covariance(row, col),
                    expected(row, col), kCovarianceTolerance);
        EXPECT_NEAR(forecast.waypoints[step].position_covariance(row, col), expected(row, col),
                    kCovarianceTolerance);
      }
    }
    if (step > 0U) {
      EXPECT_GT(forecast.waypoints[step].position_covariance(0, 0),
                forecast.waypoints[step - 1U].position_covariance(0, 0));
      EXPECT_GT(forecast.waypoints[step].position_covariance(1, 1),
                forecast.waypoints[step - 1U].position_covariance(1, 1));
    }
  }
}

TEST(KinematicTrajectoryPredictor, SingularCovarianceCanStayConstant) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 0.4;
  config.step_size_seconds = 0.2;
  config.pos_diffusion_rate = 0.0;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  prediction::TrackState track = make_track(1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
  track.covariance.setZero();
  track.covariance(4, 4) = 0.2;
  track.covariance(5, 5) = 0.2;

  const prediction::PredictedTrajectory forecast = predictor.predict(track, 0.0);
  for (const prediction::PredictedWaypoint& waypoint : forecast.waypoints) {
    EXPECT_NEAR(waypoint.position_covariance(0, 0), 0.0, kCovarianceTolerance);
    EXPECT_NEAR(waypoint.position_covariance(1, 1), 0.0, kCovarianceTolerance);
    EXPECT_NEAR(waypoint.position_covariance(0, 1), 0.0, kCovarianceTolerance);
  }
}

TEST(KinematicTrajectoryPredictor, ToleratedNegativeEigenvalueIsClamped) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 0.2;
  config.step_size_seconds = 0.2;
  config.pos_diffusion_rate = 0.0;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  prediction::TrackState track = make_track(1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
  track.covariance.setIdentity();
  track.covariance(0, 0) = -1e-12;
  const prediction::PredictedTrajectory forecast = predictor.predict(track, 10.0);
  EXPECT_GE(forecast.waypoints.front().position_covariance(0, 0), -1e-15);
  EXPECT_DOUBLE_EQ(forecast.current_state.covariance(0, 0), -1e-12);
}

TEST(KinematicTrajectoryPredictor, RejectsMalformedTracksWithoutMutation) {
  prediction::TrackState track = make_track(4, 1.0, 2.0, 3.0, 4.0, 0.5, 0.1);
  prediction::TrackState nan_state = track;
  nan_state.state(2) = std::numeric_limits<double>::quiet_NaN();
  expect_track_rejected(nan_state, 0.0, "TrackState::state");

  prediction::TrackState infinite_covariance = track;
  infinite_covariance.covariance(1, 1) = std::numeric_limits<double>::infinity();
  expect_track_rejected(infinite_covariance, 0.0, "TrackState::covariance");

  prediction::TrackState bad_size = track;
  bad_size.size.y() = 0.0;
  expect_track_rejected(bad_size, 0.0, "TrackState::size");

  prediction::TrackState asymmetric = track;
  asymmetric.covariance(0, 2) = 0.4;
  asymmetric.covariance(2, 0) = 0.0;
  expect_track_rejected(asymmetric, 0.0, "TrackState::covariance");

  prediction::TrackState indefinite = track;
  indefinite.covariance(3, 3) = -1.0;
  expect_track_rejected(indefinite, 0.0, "TrackState::covariance");

  const prediction::KinematicTrajectoryPredictor predictor;
  try {
    predictor.predict(track, std::numeric_limits<double>::quiet_NaN());
    FAIL() << "expected a nonfinite timestamp to be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("current_timestamp"), std::string::npos);
  }
}

TEST(KinematicTrajectoryPredictor, NonrepresentableForecastThrowsOverflow) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 2.0;
  config.step_size_seconds = 2.0;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  const prediction::TrackState track = make_track(1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

  EXPECT_THROW(predictor.predict(track, 1e308), std::overflow_error);

  // 1e308 * 2 is not finite. A shorter step can leave 1e308 * dt finite, and
  // DBL_MAX + 0.5 is also finite because the unit in the last place is larger
  // than the displacement.
  prediction::TrackState huge_velocity = track;
  huge_velocity.state(2) = 1e308;
  EXPECT_THROW(predictor.predict(huge_velocity, 0.0), std::overflow_error);

  prediction::PredictorConfig collapsing;
  collapsing.horizon_seconds = 0.2;
  collapsing.step_size_seconds = 0.1;
  const prediction::KinematicTrajectoryPredictor coarse(collapsing);
  EXPECT_THROW(coarse.predict(track, std::ldexp(1.0, 54)), std::overflow_error);
}

TEST(KinematicTrajectoryPredictor, TimeGridUsesFutureRoundedSteps) {
  prediction::PredictorConfig one_step;
  one_step.horizon_seconds = 0.1;
  one_step.step_size_seconds = 0.1;
  const prediction::KinematicTrajectoryPredictor minimum(one_step);
  const auto single = minimum.predict(make_track(1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), -2.5);
  ASSERT_EQ(single.waypoints.size(), 1U);
  EXPECT_NEAR(single.waypoints.front().timestamp, -2.4, kLooseTolerance);

  // 0.15 / 0.1 is not a binary half-integer, so it does not exercise std::round's
  // half-tie rule. 1.5 / 1 is exact.
  prediction::PredictorConfig half_tie;
  half_tie.horizon_seconds = 1.5;
  half_tie.step_size_seconds = 1.0;
  const prediction::KinematicTrajectoryPredictor ties(half_tie);
  EXPECT_EQ(ties.predict(make_track(1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), 0.0).waypoints.size(), 2U);

  prediction::PredictorConfig undershoot;
  undershoot.horizon_seconds = 0.26;
  undershoot.step_size_seconds = 0.1;
  const prediction::KinematicTrajectoryPredictor grid(undershoot);
  const auto samples = grid.predict(make_track(1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), 10.0);
  ASSERT_EQ(samples.waypoints.size(), 3U);
  EXPECT_NEAR(samples.waypoints.back().timestamp, 10.3, kLooseTolerance);
  for (std::size_t step = 1; step < samples.waypoints.size(); ++step) {
    EXPECT_GT(samples.waypoints[step].timestamp, samples.waypoints[step - 1U].timestamp);
  }
}

TEST(KinematicTrajectoryPredictor, StatusIsPreservedAndBatchOrderIsTransactional) {
  prediction::PredictorConfig config;
  config.horizon_seconds = 0.2;
  config.step_size_seconds = 0.1;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  const std::vector<prediction::TrackState> tracks = {
      make_track(30, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, prediction::TrackStatus::Lost),
      make_track(2, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, prediction::TrackStatus::Tentative),
      make_track(2, -1.0, 3.0, -1.0, 0.0, -0.4, 0.2, prediction::TrackStatus::Confirmed),
  };
  const std::vector<prediction::PredictedTrajectory> forecasts = predictor.predict(tracks, 4.0);
  ASSERT_EQ(forecasts.size(), tracks.size());
  for (std::size_t index = 0; index < tracks.size(); ++index) {
    EXPECT_EQ(forecasts[index].track_id, tracks[index].track_id);
    EXPECT_EQ(forecasts[index].current_state.status, tracks[index].status);
    EXPECT_EQ(forecasts[index].modes.size(), 3U);
  }

  EXPECT_TRUE(predictor.predict(std::vector<prediction::TrackState>{}, 1.0).empty());
  EXPECT_THROW(predictor.predict(std::vector<prediction::TrackState>{},
                                 std::numeric_limits<double>::infinity()),
               std::invalid_argument);

  std::vector<prediction::TrackState> rejected = tracks;
  rejected.push_back(make_track(99, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
  rejected.back().state(0) = std::numeric_limits<double>::infinity();
  try {
    predictor.predict(rejected, 4.0);
    FAIL() << "expected the later invalid track to reject the batch";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("batch index 3"), std::string::npos) << error.what();
  }
  EXPECT_DOUBLE_EQ(rejected.front().state(0), tracks.front().state(0));
}

TEST(KinematicTrajectoryPredictor, LostTrackerStateIsNotAdvancedBeforeTheForecast) {
  prediction::TrackerConfig tracker_config;
  tracker_config.min_hits_to_confirm = 1U;
  tracker_config.output_policy = prediction::TrackOutputPolicy::ConfirmedAndLost;
  prediction::MultiObjectTracker tracker(tracker_config);

  for (std::uint64_t frame = 1; frame <= 6; ++frame) {
    const double time = static_cast<double>(frame - 1U) * 0.1;
    prediction::DetectionFrame snapshot;
    snapshot.frame_id = frame;
    snapshot.timestamp = time;
    snapshot.detections = {make_detection(10.0 * time, 0.0, 0.0)};
    tracker.process_frame(snapshot);
  }
  prediction::DetectionFrame coast;
  coast.frame_id = 7;
  coast.timestamp = 0.6;
  tracker.process_frame(coast);

  const std::vector<prediction::TrackState> live =
      tracker.tracks(prediction::TrackOutputPolicy::ConfirmedAndLost);
  ASSERT_EQ(live.size(), 1U);
  EXPECT_EQ(live.front().status, prediction::TrackStatus::Lost);
  const prediction::TrackState before = live.front();

  prediction::PredictorConfig config;
  config.horizon_seconds = 0.3;
  config.step_size_seconds = 0.1;
  const prediction::KinematicTrajectoryPredictor predictor(config);
  const prediction::PredictedTrajectory forecast = predictor.predict(before, 0.6);
  EXPECT_EQ(forecast.track_id, before.track_id);
  EXPECT_EQ(forecast.current_state.status, prediction::TrackStatus::Lost);
  EXPECT_DOUBLE_EQ(forecast.current_state.state(0), before.state(0));
  EXPECT_NEAR(forecast.modes[0].waypoints.front().timestamp, 0.7, kLooseTolerance);
  EXPECT_GT(forecast.modes[0].waypoints.front().position.x(), before.state(0));
}

TEST(KinematicTrajectoryPredictor, HeaderOnlyAnchorLinksFromAnotherTranslationUnit) {
  const prediction::KinematicTrajectoryPredictor& anchor =
      prediction::test_support::predictor_linkage_anchor();
  EXPECT_DOUBLE_EQ(anchor.config().horizon_seconds, 3.0);
  const prediction::KinematicTrajectoryPredictor local;
  EXPECT_DOUBLE_EQ(anchor.config().branch_turn_rate, local.config().branch_turn_rate);
  EXPECT_EQ(&prediction::test_support::predictor_linkage_anchor(),
            &prediction::test_support::predictor_linkage_anchor());
}
