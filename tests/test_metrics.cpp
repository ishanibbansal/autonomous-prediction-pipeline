#include "prediction/metrics.hpp"
#include "prediction/predictor.hpp"

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
const TrajectoryEvaluator& metrics_linkage_anchor();
}  // namespace test_support
}  // namespace prediction

namespace {

constexpr double kScoreTolerance = 1e-12;

prediction::TrajectoryPoint make_point(double timestamp, double x, double y) {
  prediction::TrajectoryPoint point;
  point.timestamp = timestamp;
  point.position = Eigen::Vector2d(x, y);
  point.velocity = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity());
  point.heading = std::numeric_limits<double>::quiet_NaN();
  point.position_covariance =
      Eigen::Matrix2d::Constant(std::numeric_limits<double>::quiet_NaN());
  return point;
}

prediction::GroundTruthWaypoint make_truth_point(double timestamp, double x, double y) {
  prediction::GroundTruthWaypoint waypoint;
  waypoint.timestamp = timestamp;
  waypoint.position = Eigen::Vector2d(x, y);
  return waypoint;
}

prediction::PredictedTrajectory make_prediction(std::uint64_t track_id) {
  prediction::PredictedTrajectory prediction;
  prediction.track_id = track_id;
  prediction.current_state.track_id = track_id + 99;
  prediction.current_state.state.setConstant(std::numeric_limits<double>::quiet_NaN());
  prediction.current_state.covariance.setConstant(std::numeric_limits<double>::infinity());
  prediction.current_state.size.setConstant(std::numeric_limits<double>::quiet_NaN());
  prediction.waypoints.push_back(prediction::PredictedWaypoint{});
  prediction.waypoints.back().timestamp = std::numeric_limits<double>::quiet_NaN();
  prediction.waypoints.back().position.setConstant(std::numeric_limits<double>::infinity());
  prediction.waypoints.back().heading = std::numeric_limits<double>::quiet_NaN();
  return prediction;
}

prediction::GroundTruthTrajectory make_truth(std::uint64_t track_id) {
  prediction::GroundTruthTrajectory truth;
  truth.track_id = track_id;
  return truth;
}

void expect_config_rejected(const prediction::MetricsConfig& config, const char* field) {
  try {
    prediction::TrajectoryEvaluator evaluator(config);
    FAIL() << "expected std::invalid_argument naming " << field;
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(field), std::string::npos) << error.what();
  } catch (const std::exception& error) {
    FAIL() << "expected std::invalid_argument, got " << error.what();
  }
}

prediction::TrackState make_track(std::uint64_t track_id) {
  prediction::TrackState track;
  track.track_id = track_id;
  track.status = prediction::TrackStatus::Confirmed;
  track.size = Eigen::Vector2d(4.5, 1.9);
  track.state << 0.0, 0.0, 10.0, 0.0, 0.0, 0.0;
  track.covariance.setZero();
  track.covariance(0, 0) = 0.25;
  track.covariance(1, 1) = 0.25;
  track.covariance(2, 2) = 1.0;
  track.covariance(3, 3) = 1.0;
  track.covariance(4, 4) = 0.01;
  track.covariance(5, 5) = 0.01;
  return track;
}

}  // namespace

TEST(TrajectoryEvaluator, DefaultConfigAndLinkageAnchor) {
  const prediction::TrajectoryEvaluator& linked = prediction::test_support::metrics_linkage_anchor();
  const prediction::TrajectoryEvaluator local;
  EXPECT_DOUBLE_EQ(linked.config().miss_threshold_meters, 2.0);
  EXPECT_DOUBLE_EQ(linked.config().timestamp_tolerance, 1e-4);
  EXPECT_DOUBLE_EQ(local.config().miss_threshold_meters, 2.0);
  EXPECT_DOUBLE_EQ(local.config().timestamp_tolerance, 1e-4);

  prediction::MetricsConfig zero_tolerance;
  zero_tolerance.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator exact(zero_tolerance);
  EXPECT_DOUBLE_EQ(exact.config().timestamp_tolerance, 0.0);
}

TEST(TrajectoryEvaluator, RejectsInvalidConfig) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double positive_inf = std::numeric_limits<double>::infinity();
  const double negative_inf = -std::numeric_limits<double>::infinity();

  prediction::MetricsConfig config;
  config.miss_threshold_meters = nan;
  expect_config_rejected(config, "miss_threshold_meters");
  config.miss_threshold_meters = positive_inf;
  expect_config_rejected(config, "miss_threshold_meters");
  config.miss_threshold_meters = negative_inf;
  expect_config_rejected(config, "miss_threshold_meters");
  config.miss_threshold_meters = 0.0;
  expect_config_rejected(config, "miss_threshold_meters");
  config.miss_threshold_meters = -1.0;
  expect_config_rejected(config, "miss_threshold_meters");

  config = {};
  config.timestamp_tolerance = nan;
  expect_config_rejected(config, "timestamp_tolerance");
  config.timestamp_tolerance = positive_inf;
  expect_config_rejected(config, "timestamp_tolerance");
  config.timestamp_tolerance = negative_inf;
  expect_config_rejected(config, "timestamp_tolerance");
  config.timestamp_tolerance = -1e-12;
  expect_config_rejected(config, "timestamp_tolerance");
}

TEST(TrajectoryEvaluator, IndependentMinimaPreferEarliestMode) {
  prediction::MetricsConfig config;
  config.miss_threshold_meters = 2.0;
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);

  prediction::PredictedTrajectory prediction = make_prediction(7);
  prediction::TrajectoryMode mode0;
  mode0.probability = std::numeric_limits<double>::quiet_NaN();
  mode0.waypoints = {make_point(1.0, 0.0, 0.0), make_point(2.0, 2.0, 0.0)};
  prediction::TrajectoryMode mode1;
  mode1.probability = -4.0;
  mode1.waypoints = {make_point(1.0, 4.0, 0.0), make_point(2.0, 0.0, 0.0)};
  prediction.modes = {mode0, mode1};

  prediction::GroundTruthTrajectory truth = make_truth(7);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0), make_truth_point(2.0, 0.0, 0.0)};

  const prediction::TrajectoryMetrics metrics = evaluator.evaluate(prediction, truth);
  EXPECT_EQ(metrics.track_id, 7U);
  EXPECT_DOUBLE_EQ(metrics.min_ade, 1.0);
  EXPECT_DOUBLE_EQ(metrics.min_fde, 0.0);
  EXPECT_EQ(metrics.best_ade_mode, 0U);
  EXPECT_EQ(metrics.best_fde_mode, 1U);
  EXPECT_FALSE(metrics.is_miss);
}

TEST(TrajectoryEvaluator, ThreeFourFiveAndSinglePoint) {
  const prediction::TrajectoryEvaluator evaluator;

  prediction::PredictedTrajectory prediction = make_prediction(1);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(0.5, 3.0, 4.0)};
  prediction.modes = {mode};

  prediction::GroundTruthTrajectory truth = make_truth(1);
  truth.waypoints = {make_truth_point(0.5, 0.0, 0.0)};

  const prediction::TrajectoryMetrics metrics = evaluator.evaluate(prediction, truth);
  EXPECT_DOUBLE_EQ(metrics.min_ade, 5.0);
  EXPECT_DOUBLE_EQ(metrics.min_fde, 5.0);
  EXPECT_EQ(metrics.best_ade_mode, 0U);
  EXPECT_EQ(metrics.best_fde_mode, 0U);
  EXPECT_TRUE(metrics.is_miss);
}

TEST(TrajectoryEvaluator, ExactTiesKeepEarliestMode) {
  prediction::MetricsConfig config;
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);

  prediction::PredictedTrajectory prediction = make_prediction(3);
  prediction::TrajectoryMode first;
  first.waypoints = {make_point(-2.0, 1.0, 0.0), make_point(-1.0, 0.0, 2.0)};
  prediction::TrajectoryMode second = first;
  prediction::TrajectoryMode third;
  third.waypoints = {make_point(-2.0, 0.0, 0.0), make_point(-1.0, 0.0, 0.0)};
  prediction.modes = {first, second, third};

  prediction::GroundTruthTrajectory truth = make_truth(3);
  truth.waypoints = {make_truth_point(-2.0, 0.0, 0.0), make_truth_point(-1.0, 0.0, 0.0)};

  const prediction::TrajectoryMetrics metrics = evaluator.evaluate(prediction, truth);
  EXPECT_DOUBLE_EQ(metrics.min_ade, 0.0);
  EXPECT_DOUBLE_EQ(metrics.min_fde, 0.0);
  EXPECT_EQ(metrics.best_ade_mode, 2U);
  EXPECT_EQ(metrics.best_fde_mode, 2U);

  prediction.modes = {first, second};
  const prediction::TrajectoryMetrics tied = evaluator.evaluate(prediction, truth);
  EXPECT_DOUBLE_EQ(tied.min_ade, 1.5);
  EXPECT_DOUBLE_EQ(tied.min_fde, 2.0);
  EXPECT_EQ(tied.best_ade_mode, 0U);
  EXPECT_EQ(tied.best_fde_mode, 0U);
  EXPECT_FALSE(tied.is_miss);
}

TEST(TrajectoryEvaluator, MissThresholdEqualityIsAHit) {
  prediction::MetricsConfig config;
  config.miss_threshold_meters = 2.0;
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);

  prediction::PredictedTrajectory prediction = make_prediction(4);
  prediction::TrajectoryMode equal;
  equal.waypoints = {make_point(1.0, 0.0, 0.0), make_point(2.0, 0.0, 2.0)};
  prediction.modes = {equal};
  prediction::GroundTruthTrajectory truth = make_truth(4);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0), make_truth_point(2.0, 0.0, 0.0)};

  const prediction::TrajectoryMetrics hit = evaluator.evaluate(prediction, truth);
  EXPECT_DOUBLE_EQ(hit.min_fde, 2.0);
  EXPECT_FALSE(hit.is_miss);

  prediction.modes[0].waypoints[1].position.y() = std::nextafter(2.0, 3.0);
  const prediction::TrajectoryMetrics miss = evaluator.evaluate(prediction, truth);
  EXPECT_GT(miss.min_fde, config.miss_threshold_meters);
  EXPECT_TRUE(miss.is_miss);
}

TEST(TrajectoryEvaluator, RejectsEmptyAndMismatchedStructure) {
  const prediction::TrajectoryEvaluator evaluator;
  prediction::PredictedTrajectory prediction = make_prediction(8);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(1.0, 0.0, 0.0)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(8);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0)};

  prediction::GroundTruthTrajectory empty_truth = truth;
  empty_truth.waypoints.clear();
  EXPECT_THROW(evaluator.evaluate(prediction, empty_truth), std::invalid_argument);

  prediction::PredictedTrajectory legacy_only = prediction;
  legacy_only.modes.clear();
  EXPECT_FALSE(legacy_only.waypoints.empty());
  EXPECT_THROW(evaluator.evaluate(legacy_only, truth), std::invalid_argument);

  prediction::PredictedTrajectory wrong_id = prediction;
  wrong_id.track_id = 9;
  EXPECT_THROW(evaluator.evaluate(wrong_id, truth), std::invalid_argument);

  prediction::PredictedTrajectory short_mode = prediction;
  short_mode.modes.push_back(prediction::TrajectoryMode{});
  EXPECT_THROW(evaluator.evaluate(short_mode, truth), std::invalid_argument);

  prediction::PredictedTrajectory long_mode = prediction;
  long_mode.modes[0].waypoints.push_back(make_point(2.0, 0.0, 0.0));
  EXPECT_THROW(evaluator.evaluate(long_mode, truth), std::invalid_argument);
}

TEST(TrajectoryEvaluator, RejectsNonFiniteUsedFieldsButIgnoresUnusedCorruption) {
  const prediction::TrajectoryEvaluator evaluator;
  prediction::PredictedTrajectory prediction = make_prediction(11);
  prediction::TrajectoryMode mode;
  mode.probability = std::numeric_limits<double>::quiet_NaN();
  mode.waypoints = {make_point(1.0, 3.0, 4.0)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(11);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0)};

  const prediction::TrajectoryMetrics scored = evaluator.evaluate(prediction, truth);
  EXPECT_DOUBLE_EQ(scored.min_ade, 5.0);

  prediction.modes[0].waypoints[0].position.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
  prediction.modes[0].waypoints[0].position.x() = 3.0;
  prediction.modes[0].waypoints[0].position.y() = std::numeric_limits<double>::infinity();
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
  prediction.modes[0].waypoints[0].position.y() = 4.0;
  prediction.modes[0].waypoints[0].timestamp = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);

  prediction.modes[0].waypoints[0] = make_point(1.0, 3.0, 4.0);
  truth.waypoints[0].position.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
  truth.waypoints[0].position.x() = 0.0;
  truth.waypoints[0].position.y() = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
  truth.waypoints[0].position.y() = 0.0;
  truth.waypoints[0].timestamp = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
}

TEST(TrajectoryEvaluator, MalformedLaterModeInvalidatesPerfectEarlierMode) {
  prediction::MetricsConfig config;
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);

  prediction::PredictedTrajectory prediction = make_prediction(12);
  prediction::TrajectoryMode perfect;
  perfect.waypoints = {make_point(1.0, 0.0, 0.0), make_point(2.0, 0.0, 0.0)};
  prediction::TrajectoryMode broken;
  broken.waypoints = {make_point(1.0, 0.0, 0.0), make_point(2.0, std::numeric_limits<double>::quiet_NaN(), 0.0)};
  prediction.modes = {perfect, broken};

  prediction::GroundTruthTrajectory truth = make_truth(12);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0), make_truth_point(2.0, 0.0, 0.0)};
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
}

TEST(TrajectoryEvaluator, TimestampToleranceIsInclusiveAbsolute) {
  prediction::MetricsConfig exact;
  exact.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator zero(exact);

  prediction::PredictedTrajectory prediction = make_prediction(15);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(0.0, 0.0, 0.0), make_point(1.0, 0.0, 0.0)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(15);
  truth.waypoints = {make_truth_point(0.0, 0.0, 0.0), make_truth_point(1.0, 0.0, 0.0)};
  EXPECT_DOUBLE_EQ(zero.evaluate(prediction, truth).min_fde, 0.0);

  const double tolerance = 0.25;
  const double boundary = 1.0 + tolerance;
  prediction::MetricsConfig config;
  config.timestamp_tolerance = tolerance;
  const prediction::TrajectoryEvaluator evaluator(config);
  prediction.modes[0].waypoints[1].timestamp = boundary;
  EXPECT_DOUBLE_EQ(std::abs(boundary - 1.0), tolerance);
  EXPECT_DOUBLE_EQ(evaluator.evaluate(prediction, truth).min_ade, 0.0);

  prediction.modes[0].waypoints[1].timestamp = std::nextafter(boundary, 2.0);
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
  prediction.modes[0].waypoints[1].timestamp = std::nextafter(1.0 - tolerance, 0.0);
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
}

TEST(TrajectoryEvaluator, RejectsNonIncreasingTimesEvenWhenAlignedIndividually) {
  prediction::MetricsConfig config;
  config.timestamp_tolerance = 1.0;
  const prediction::TrajectoryEvaluator evaluator(config);

  prediction::PredictedTrajectory prediction = make_prediction(16);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(1.0, 0.0, 0.0), make_point(2.0, 0.0, 0.0)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(16);
  truth.waypoints = {make_truth_point(2.0, 0.0, 0.0), make_truth_point(2.0, 1.0, 0.0)};
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);

  truth.waypoints[1].timestamp = 1.5;
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);

  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0), make_truth_point(2.0, 0.0, 0.0)};
  prediction.modes[0].waypoints[1].timestamp = 1.0;
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
  prediction.modes[0].waypoints[0].timestamp = 1.6;
  prediction.modes[0].waypoints[1].timestamp = 1.2;
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
}

TEST(TrajectoryEvaluator, ExtremeOppositeTimestampsAreMisaligned) {
  prediction::MetricsConfig config;
  config.timestamp_tolerance = std::numeric_limits<double>::max();
  const prediction::TrajectoryEvaluator evaluator(config);

  prediction::PredictedTrajectory prediction = make_prediction(17);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(std::numeric_limits<double>::max(), 0.0, 0.0)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(17);
  truth.waypoints = {make_truth_point(-std::numeric_limits<double>::max(), 0.0, 0.0)};
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::invalid_argument);
}

TEST(TrajectoryEvaluator, LargeRepresentableHypotIsNotOverflow) {
  prediction::MetricsConfig config;
  config.miss_threshold_meters = std::numeric_limits<double>::max();
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);

  // Large enough that a naive dx*dx + dy*dy overflows, but still inside the
  // range where std::hypot returns a finite norm. This is not max/2: that
  // subtraction is finite, yet its hypotenuse is not.
  const double large = std::ldexp(1.0, 1022);
  prediction::PredictedTrajectory prediction = make_prediction(20);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(1.0, large, -large)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(20);
  truth.waypoints = {make_truth_point(1.0, -large, large)};

  const double dx = large - (-large);
  const double dy = -large - large;
  ASSERT_FALSE(std::isfinite(dx * dx + dy * dy));
  ASSERT_TRUE(std::isfinite(dx));
  ASSERT_TRUE(std::isfinite(dy));
  const double expected = std::hypot(dx, dy);
  ASSERT_TRUE(std::isfinite(expected));
  const prediction::TrajectoryMetrics metrics = evaluator.evaluate(prediction, truth);
  EXPECT_DOUBLE_EQ(metrics.min_ade, expected);
  EXPECT_DOUBLE_EQ(metrics.min_fde, expected);
  EXPECT_FALSE(metrics.is_miss);
}

TEST(TrajectoryEvaluator, CoordinateSubtractionOverflowThrows) {
  const prediction::TrajectoryEvaluator evaluator;
  prediction::PredictedTrajectory prediction = make_prediction(21);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(1.0, std::numeric_limits<double>::max(), 0.0)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(21);
  truth.waypoints = {make_truth_point(1.0, -std::numeric_limits<double>::max(), 0.0)};
  ASSERT_FALSE(std::isfinite(std::numeric_limits<double>::max() - (-std::numeric_limits<double>::max())));
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::overflow_error);
}

TEST(TrajectoryEvaluator, NormOverflowThrows) {
  const prediction::TrajectoryEvaluator evaluator;
  const double ordinate = std::nextafter(std::numeric_limits<double>::max(), 0.0);
  prediction::PredictedTrajectory prediction = make_prediction(22);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(1.0, ordinate, ordinate)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(22);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0)};

  ASSERT_TRUE(std::isfinite(ordinate - 0.0));
  ASSERT_FALSE(std::isfinite(std::hypot(ordinate, ordinate)));
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::overflow_error);
}

TEST(TrajectoryEvaluator, PerModeSumOverflowThrowsEvenForLosingMode) {
  prediction::MetricsConfig config;
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);
  const double huge = std::numeric_limits<double>::max();

  prediction::PredictedTrajectory prediction = make_prediction(23);
  prediction::TrajectoryMode perfect;
  perfect.waypoints = {make_point(1.0, 0.0, 0.0), make_point(2.0, 0.0, 0.0)};
  prediction::TrajectoryMode overflowing;
  overflowing.waypoints = {make_point(1.0, huge, 0.0), make_point(2.0, huge, 0.0)};
  prediction.modes = {perfect, overflowing};

  prediction::GroundTruthTrajectory truth = make_truth(23);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0), make_truth_point(2.0, 0.0, 0.0)};
  ASSERT_TRUE(std::isfinite(std::hypot(huge, 0.0)));
  ASSERT_FALSE(std::isfinite(huge + huge));
  EXPECT_THROW(evaluator.evaluate(prediction, truth), std::overflow_error);
}

TEST(TrajectoryEvaluator, BatchPreservesOrderAndWeightsPairsEqually) {
  prediction::MetricsConfig config;
  config.miss_threshold_meters = 2.0;
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);

  auto pair = [](std::uint64_t id, double error, std::size_t count) {
    prediction::PredictedTrajectory prediction = make_prediction(id);
    prediction::GroundTruthTrajectory truth = make_truth(id);
    prediction::TrajectoryMode mode;
    for (std::size_t index = 0; index < count; ++index) {
      const double timestamp = static_cast<double>(index);
      mode.waypoints.push_back(make_point(timestamp, error, 0.0));
      truth.waypoints.push_back(make_truth_point(timestamp, 0.0, 0.0));
    }
    prediction.modes = {mode};
    return std::make_pair(prediction, truth);
  };

  const auto first = pair(9, 0.0, 1);
  const auto second = pair(2, 1.0, 4);
  const auto third = pair(9, 4.0, 2);
  const std::vector<prediction::PredictedTrajectory> predictions = {first.first, second.first, third.first};
  const std::vector<prediction::GroundTruthTrajectory> truths = {first.second, second.second, third.second};

  const prediction::BatchMetricsSummary summary = evaluator.evaluate_batch(predictions, truths);
  ASSERT_EQ(summary.evaluated_count, 3U);
  ASSERT_EQ(summary.per_track.size(), 3U);
  EXPECT_EQ(summary.per_track[0].track_id, 9U);
  EXPECT_EQ(summary.per_track[1].track_id, 2U);
  EXPECT_EQ(summary.per_track[2].track_id, 9U);
  EXPECT_DOUBLE_EQ(summary.per_track[0].min_ade, 0.0);
  EXPECT_DOUBLE_EQ(summary.per_track[1].min_ade, 1.0);
  EXPECT_DOUBLE_EQ(summary.per_track[2].min_fde, 4.0);
  EXPECT_FALSE(summary.per_track[0].is_miss);
  EXPECT_FALSE(summary.per_track[1].is_miss);
  EXPECT_TRUE(summary.per_track[2].is_miss);
  EXPECT_NEAR(summary.mean_min_ade, (0.0 + 1.0 + 4.0) / 3.0, kScoreTolerance);
  EXPECT_NEAR(summary.mean_min_fde, (0.0 + 1.0 + 4.0) / 3.0, kScoreTolerance);
  EXPECT_DOUBLE_EQ(summary.miss_rate, 1.0 / 3.0);
}

TEST(TrajectoryEvaluator, CrossedIdsAreRejectedRatherThanReordered) {
  const prediction::TrajectoryEvaluator evaluator;
  prediction::PredictedTrajectory left = make_prediction(1);
  prediction::PredictedTrajectory right = make_prediction(2);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(1.0, 0.0, 0.0)};
  left.modes = {mode};
  right.modes = {mode};
  prediction::GroundTruthTrajectory truth1 = make_truth(1);
  prediction::GroundTruthTrajectory truth2 = make_truth(2);
  truth1.waypoints = {make_truth_point(1.0, 0.0, 0.0)};
  truth2.waypoints = {make_truth_point(1.0, 0.0, 0.0)};

  EXPECT_THROW(evaluator.evaluate_batch({left, right}, {truth2, truth1}), std::invalid_argument);
}

TEST(TrajectoryEvaluator, BatchRejectsEmptyUnequalAndLaterInvalidPairs) {
  const prediction::TrajectoryEvaluator evaluator;
  prediction::PredictedTrajectory prediction = make_prediction(1);
  prediction::TrajectoryMode mode;
  mode.waypoints = {make_point(1.0, 0.0, 0.0)};
  prediction.modes = {mode};
  prediction::GroundTruthTrajectory truth = make_truth(1);
  truth.waypoints = {make_truth_point(1.0, 0.0, 0.0)};

  EXPECT_THROW(evaluator.evaluate_batch({}, {}), std::invalid_argument);
  EXPECT_THROW(evaluator.evaluate_batch({prediction}, {}), std::invalid_argument);
  EXPECT_THROW(evaluator.evaluate_batch({}, {truth}), std::invalid_argument);

  prediction::GroundTruthTrajectory bad = truth;
  bad.waypoints[0].position.x() = std::numeric_limits<double>::quiet_NaN();
  const std::vector<prediction::PredictedTrajectory> predictions = {prediction, prediction};
  const std::vector<prediction::GroundTruthTrajectory> truths = {truth, bad};
  EXPECT_THROW(evaluator.evaluate_batch(predictions, truths), std::invalid_argument);
  EXPECT_DOUBLE_EQ(predictions[1].modes[0].waypoints[0].position.x(), 0.0);
  EXPECT_TRUE(std::isnan(truths[1].waypoints[0].position.x()));
}

TEST(TrajectoryEvaluator, BatchSumOverflowRejectsTheCall) {
  prediction::MetricsConfig config;
  config.miss_threshold_meters = std::numeric_limits<double>::max();
  config.timestamp_tolerance = 0.0;
  const prediction::TrajectoryEvaluator evaluator(config);
  const double huge = std::numeric_limits<double>::max();

  auto scored = [&](std::uint64_t id, double error) {
    prediction::PredictedTrajectory prediction = make_prediction(id);
    prediction::TrajectoryMode mode;
    mode.waypoints = {make_point(1.0, error, 0.0)};
    prediction.modes = {mode};
    prediction::GroundTruthTrajectory truth = make_truth(id);
    truth.waypoints = {make_truth_point(1.0, 0.0, 0.0)};
    return std::make_pair(prediction, truth);
  };
  const auto first = scored(1, huge);
  const auto second = scored(2, huge);
  ASSERT_TRUE(std::isfinite(evaluator.evaluate(first.first, first.second).min_ade));
  EXPECT_THROW(evaluator.evaluate_batch({first.first, second.first}, {first.second, second.second}),
               std::overflow_error);
}

TEST(TrajectoryEvaluator, ScoresAnActualKinematicForecast) {
  prediction::PredictorConfig predictor_config;
  predictor_config.horizon_seconds = 1.0;
  predictor_config.step_size_seconds = 0.5;
  predictor_config.branch_turn_rate = 0.2;
  predictor_config.prob_straight = 0.5;
  predictor_config.prob_left = 0.25;
  predictor_config.prob_right = 0.25;
  predictor_config.pos_diffusion_rate = 0.0;
  const prediction::KinematicTrajectoryPredictor predictor(predictor_config);
  const prediction::TrackState track = make_track(42);
  const prediction::PredictedTrajectory forecast = predictor.predict(track, 10.0);
  ASSERT_EQ(forecast.modes.size(), 3U);
  ASSERT_FALSE(forecast.modes[0].waypoints.empty());

  prediction::GroundTruthTrajectory truth = make_truth(forecast.track_id);
  truth.waypoints.reserve(forecast.modes[0].waypoints.size());
  for (const prediction::TrajectoryPoint& point : forecast.modes[0].waypoints) {
    truth.waypoints.push_back(make_truth_point(point.timestamp, point.position.x(), point.position.y()));
  }

  prediction::MetricsConfig config;
  config.timestamp_tolerance = 0.0;
  config.miss_threshold_meters = 1.0;
  const prediction::TrajectoryEvaluator evaluator(config);
  const prediction::TrajectoryMetrics metrics = evaluator.evaluate(forecast, truth);
  EXPECT_EQ(metrics.track_id, 42U);
  EXPECT_EQ(metrics.best_ade_mode, 0U);
  EXPECT_EQ(metrics.best_fde_mode, 0U);
  EXPECT_NEAR(metrics.min_ade, 0.0, kScoreTolerance);
  EXPECT_NEAR(metrics.min_fde, 0.0, kScoreTolerance);
  EXPECT_FALSE(metrics.is_miss);
  EXPECT_GT(forecast.modes[1].waypoints.back().position.norm(), 0.0);
}
