#include "prediction/data_association.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace prediction {
namespace test_support {
const DataAssociation& data_association_linkage_anchor();
}  // namespace test_support
}  // namespace prediction

namespace {

prediction::Detection detection_at(double x, double y, double heading = 0.0) {
  prediction::Detection detection;
  detection.position = Eigen::Vector2d(x, y);
  detection.size = Eigen::Vector2d(4.5, 1.8);
  detection.heading = heading;
  detection.score = 0.8;
  return detection;
}

prediction::TrackState track_at(double x, double y, double heading, double position_variance,
                                double heading_variance, std::uint64_t id) {
  prediction::TrackState track;
  track.track_id = id;
  track.status = prediction::TrackStatus::Confirmed;
  track.size = Eigen::Vector2d(4.5, 1.8);
  track.state.setZero();
  track.state(0) = x;
  track.state(1) = y;
  track.state(4) = heading;
  track.covariance.setZero();
  track.covariance(0, 0) = position_variance;
  track.covariance(1, 1) = position_variance;
  track.covariance(2, 2) = 1.0;
  track.covariance(3, 3) = 1.0;
  track.covariance(4, 4) = heading_variance;
  track.covariance(5, 5) = 1.0;
  return track;
}

prediction::KalmanFilter tight_filter() {
  prediction::KalmanFilterConfig config;
  config.sigma_pos_sq = 0.01;
  config.sigma_psi_sq = 0.01;
  return prediction::KalmanFilter{config};
}

prediction::CostMatrix matrix_from(
    std::size_t tracks, std::size_t detections,
    const std::vector<std::optional<double>>& costs) {
  prediction::CostMatrix matrix;
  matrix.track_count = tracks;
  matrix.detection_count = detections;
  matrix.costs = costs;
  return matrix;
}

void expect_partition(const prediction::AssociationResult& result, std::size_t track_count,
                      std::size_t detection_count) {
  std::vector<int> tracks(track_count, 0);
  std::vector<int> detections(detection_count, 0);
  std::size_t previous_track = 0;
  for (std::size_t index = 0; index < result.matches.size(); ++index) {
    const prediction::TrackDetectionMatch& match = result.matches[index];
    ASSERT_LT(match.track_index, track_count);
    ASSERT_LT(match.detection_index, detection_count);
    if (index > 0) {
      EXPECT_LT(previous_track, match.track_index);
    }
    previous_track = match.track_index;
    EXPECT_EQ(tracks[match.track_index], 0);
    EXPECT_EQ(detections[match.detection_index], 0);
    tracks[match.track_index] = 1;
    detections[match.detection_index] = 1;
  }
  std::size_t previous = 0;
  for (std::size_t index = 0; index < result.unmatched_tracks.size(); ++index) {
    const std::size_t track = result.unmatched_tracks[index];
    ASSERT_LT(track, track_count);
    if (index > 0) {
      EXPECT_LT(previous, track);
    }
    previous = track;
    EXPECT_EQ(tracks[track], 0);
    tracks[track] = 1;
  }
  previous = 0;
  for (std::size_t index = 0; index < result.unmatched_detections.size(); ++index) {
    const std::size_t detection = result.unmatched_detections[index];
    ASSERT_LT(detection, detection_count);
    if (index > 0) {
      EXPECT_LT(previous, detection);
    }
    previous = detection;
    EXPECT_EQ(detections[detection], 0);
    detections[detection] = 1;
  }
  EXPECT_EQ(static_cast<std::size_t>(std::count(tracks.begin(), tracks.end(), 1)), track_count);
  EXPECT_EQ(static_cast<std::size_t>(std::count(detections.begin(), detections.end(), 1)),
            detection_count);
}

}  // namespace

TEST(DataAssociation, ConfigRejectsNonPositiveAndNonFiniteThreshold) {
  EXPECT_NO_THROW(prediction::DataAssociation{});
  const double invalid[] = {0.0, -0.25, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity()};
  for (const double threshold : invalid) {
    prediction::DataAssociationConfig config;
    config.gate_threshold_sq = threshold;
    EXPECT_THROW(prediction::DataAssociation{config}, std::invalid_argument);
  }
  prediction::DataAssociationConfig config;
  config.gate_threshold_sq = 1.5;
  const prediction::DataAssociation associator(config);
  EXPECT_DOUBLE_EQ(associator.config().gate_threshold_sq, 1.5);
}

TEST(DataAssociation, AntiGreedyMatrixChoosesLowerTotalCost) {
  // Locally, row 0 prefers column 0 (cost 0.1 versus 0.4). Taking that pair
  // forces row 1 onto cost 0.9, for a total of 1.0. The global assignment is
  // the crossed pairing at cost 0.6.
  const prediction::CostMatrix costs = matrix_from(2, 2, {0.1, 0.4, 0.2, 0.9});
  const std::vector<std::optional<std::size_t>> assigned =
      prediction::DataAssociation::solve_cost_matrix(costs);
  ASSERT_EQ(assigned.size(), 2U);
  ASSERT_TRUE(assigned[0].has_value());
  ASSERT_TRUE(assigned[1].has_value());
  EXPECT_EQ(*assigned[0], 1U);
  EXPECT_EQ(*assigned[1], 0U);
}

TEST(DataAssociation, CardinalityOutranksSeveralCheaperEdges) {
  // Two cheap disjoint edges sum to 0.3. Replacing both is required to use the
  // third detection, whose three normalized edges cost 0.9. Maximum cardinality
  // is three, so all three detections must be selected despite the larger sum.
  std::vector<std::optional<double>> values(3U * 3U, std::nullopt);
  values[0 * 3U + 0U] = 0.1;
  values[1 * 3U + 1U] = 0.2;
  values[0 * 3U + 2U] = 0.9;
  values[1 * 3U + 2U] = 0.9;
  values[2 * 3U + 2U] = 0.9;
  values[2 * 3U + 0U] = 0.9;
  values[2 * 3U + 1U] = 0.9;

  const std::vector<std::optional<std::size_t>> assigned =
      prediction::DataAssociation::solve_cost_matrix(matrix_from(3, 3, values));
  ASSERT_EQ(assigned.size(), 3U);
  std::vector<int> used(3, 0);
  for (const std::optional<std::size_t>& column : assigned) {
    ASSERT_TRUE(column.has_value());
    ASSERT_LT(*column, 3U);
    EXPECT_EQ(used[*column], 0);
    used[*column] = 1;
  }
  EXPECT_EQ(used[0] + used[1] + used[2], 3);
}

TEST(DataAssociation, HardGateBoundaryAndUnmatchedPartition) {
  const prediction::KalmanFilter filter = tight_filter();
  prediction::DataAssociationConfig config;
  config.gate_threshold_sq = 1.0;
  const prediction::DataAssociation associator(config);

  const double variance = 1.0 - config.gate_threshold_sq * filter.config().sigma_pos_sq;
  ASSERT_GT(variance, 0.0);
  std::vector<prediction::TrackState> tracks = {
      track_at(0.0, 0.0, 0.0, variance, 0.01, 1),
      track_at(10.0, 0.0, 0.0, variance, 0.01, 2),
  };
  std::vector<prediction::Detection> detections = {
      detection_at(1.0, 0.0),
      detection_at(10.0, 2.0),
  };

  const std::optional<double> accepted =
      filter.mahalanobis_distance_squared(tracks[0], detections[0]);
  const std::optional<double> rejected =
      filter.mahalanobis_distance_squared(tracks[1], detections[1]);
  ASSERT_TRUE(accepted.has_value());
  ASSERT_TRUE(rejected.has_value());
  EXPECT_NEAR(*accepted, 1.0, 1e-9);
  EXPECT_GT(*rejected, config.gate_threshold_sq);

  const prediction::AssociationResult result = associator.associate(tracks, detections, filter);
  expect_partition(result, 2, 2);
  ASSERT_EQ(result.matches.size(), 1U);
  EXPECT_EQ(result.matches[0].track_index, 0U);
  EXPECT_EQ(result.matches[0].detection_index, 0U);
  EXPECT_NEAR(result.matches[0].mahalanobis_distance_sq, *accepted, 1e-12);
  ASSERT_EQ(result.unmatched_tracks.size(), 1U);
  EXPECT_EQ(result.unmatched_tracks[0], 1U);
  ASSERT_EQ(result.unmatched_detections.size(), 1U);
  EXPECT_EQ(result.unmatched_detections[0], 1U);
}

TEST(DataAssociation, RectangularEmptyAndFullyGatedInputs) {
  const prediction::KalmanFilter filter = tight_filter();
  const prediction::DataAssociation associator;

  const prediction::AssociationResult no_tracks =
      associator.associate({}, {detection_at(1.0, 1.0), detection_at(2.0, 2.0)}, filter);
  expect_partition(no_tracks, 0, 2);
  EXPECT_TRUE(no_tracks.matches.empty());
  EXPECT_TRUE(no_tracks.unmatched_tracks.empty());
  ASSERT_EQ(no_tracks.unmatched_detections.size(), 2U);
  EXPECT_EQ(no_tracks.unmatched_detections[0], 0U);
  EXPECT_EQ(no_tracks.unmatched_detections[1], 1U);

  const prediction::AssociationResult no_detections =
      associator.associate({track_at(0.0, 0.0, 0.0, 1.0, 0.1, 7)}, {}, filter);
  expect_partition(no_detections, 1, 0);
  EXPECT_TRUE(no_detections.matches.empty());
  ASSERT_EQ(no_detections.unmatched_tracks.size(), 1U);
  EXPECT_EQ(no_detections.unmatched_tracks[0], 0U);

  const prediction::AssociationResult empty = associator.associate({}, {}, filter);
  expect_partition(empty, 0, 0);
  EXPECT_TRUE(empty.matches.empty());

  std::vector<prediction::TrackState> tracks = {
      track_at(0.0, 0.0, 0.0, 1.0, 0.05, 1),
      track_at(5.0, 0.0, 0.0, 1.0, 0.05, 2),
      track_at(-5.0, 0.0, 0.0, 1.0, 0.05, 3),
  };
  std::vector<prediction::Detection> detections = {
      detection_at(0.1, -0.05),
      detection_at(5.1, 0.02),
  };
  const prediction::AssociationResult wide = associator.associate(tracks, detections, filter);
  expect_partition(wide, 3, 2);
  ASSERT_EQ(wide.matches.size(), 2U);
  EXPECT_EQ(wide.matches[0].track_index, 0U);
  EXPECT_EQ(wide.matches[1].track_index, 1U);
  EXPECT_EQ(wide.unmatched_tracks, std::vector<std::size_t>{2});
  EXPECT_TRUE(wide.unmatched_detections.empty());

  std::vector<prediction::Detection> crowded = detections;
  crowded.push_back(detection_at(-4.9, 0.1));
  crowded.push_back(detection_at(80.0, 80.0));
  const prediction::AssociationResult tall = associator.associate(
      {tracks[0], tracks[1]}, crowded, filter);
  expect_partition(tall, 2, 4);
  ASSERT_EQ(tall.matches.size(), 2U);
  EXPECT_EQ(tall.matches[0].detection_index, 0U);
  EXPECT_EQ(tall.matches[1].detection_index, 1U);
  EXPECT_EQ(tall.unmatched_detections, (std::vector<std::size_t>{2, 3}));

  prediction::DataAssociationConfig closed_gate;
  closed_gate.gate_threshold_sq = 1e-12;
  const prediction::DataAssociation closed(closed_gate);
  const prediction::AssociationResult none =
      closed.associate({tracks[0]}, {detection_at(1.0, 0.0)}, filter);
  expect_partition(none, 1, 1);
  EXPECT_TRUE(none.matches.empty());
  EXPECT_EQ(none.unmatched_tracks, std::vector<std::size_t>{0});
  EXPECT_EQ(none.unmatched_detections, std::vector<std::size_t>{0});
}

TEST(DataAssociation, InadmissibleEntriesStayUnmatchedWithoutDisturbingOthers) {
  const prediction::KalmanFilter filter = tight_filter();
  const prediction::DataAssociation associator;

  prediction::TrackState bad_track = track_at(0.0, 0.0, 0.0, 1.0, 0.1, 9);
  bad_track.size.setZero();
  prediction::Detection bad_detection = detection_at(2.0, 0.0);
  bad_detection.size.y() = 0.0;
  prediction::TrackState indefinite = track_at(2.0, 0.0, 0.0, 1.0, 0.1, 10);
  indefinite.covariance(0, 2) = 20.0;
  indefinite.covariance(2, 0) = 20.0;

  std::vector<prediction::TrackState> tracks = {
      bad_track,
      track_at(2.0, 0.0, 0.0, 0.25, 0.05, 11),
      indefinite,
  };
  std::vector<prediction::Detection> detections = {
      bad_detection,
      detection_at(2.05, -0.02),
  };
  const prediction::TrackState valid_before = tracks[1];
  const prediction::Detection valid_detection_before = detections[1];

  const prediction::AssociationResult result = associator.associate(tracks, detections, filter);
  expect_partition(result, 3, 2);
  ASSERT_EQ(result.matches.size(), 1U);
  EXPECT_EQ(result.matches[0].track_index, 1U);
  EXPECT_EQ(result.matches[0].detection_index, 1U);
  EXPECT_LT(result.matches[0].mahalanobis_distance_sq, associator.config().gate_threshold_sq);
  EXPECT_EQ(result.unmatched_tracks, (std::vector<std::size_t>{0, 2}));
  EXPECT_EQ(result.unmatched_detections, std::vector<std::size_t>{0});
  EXPECT_EQ(tracks[1].track_id, valid_before.track_id);
  EXPECT_EQ(tracks[1].state, valid_before.state);
  EXPECT_EQ(detections[1].position, valid_detection_before.position);
}

TEST(DataAssociation, WrappedHeadingUsesFilterResidual) {
  const prediction::KalmanFilter filter = tight_filter();
  const prediction::DataAssociation associator;
  const prediction::TrackState track = track_at(0.0, 0.0, prediction::kPi - 0.02, 0.25, 0.05, 4);
  const prediction::Detection wrapped = detection_at(0.0, 0.0, -prediction::kPi + 0.03);
  const std::optional<double> expected = filter.mahalanobis_distance_squared(track, wrapped);
  ASSERT_TRUE(expected.has_value());
  EXPECT_LT(*expected, associator.config().gate_threshold_sq);

  const prediction::AssociationResult result = associator.associate({track}, {wrapped}, filter);
  expect_partition(result, 1, 1);
  ASSERT_EQ(result.matches.size(), 1U);
  EXPECT_NEAR(result.matches[0].mahalanobis_distance_sq, *expected, 1e-12);
}

TEST(DataAssociation, EqualCostsFollowStableIndexOrder) {
  const prediction::CostMatrix costs = matrix_from(2, 2, {0.5, 0.5, 0.5, 0.5});
  const std::vector<std::optional<std::size_t>> first =
      prediction::DataAssociation::solve_cost_matrix(costs);
  const std::vector<std::optional<std::size_t>> second =
      prediction::DataAssociation::solve_cost_matrix(costs);
  ASSERT_EQ(first.size(), 2U);
  ASSERT_TRUE(first[0].has_value());
  ASSERT_TRUE(first[1].has_value());
  EXPECT_NE(*first[0], *first[1]);
  EXPECT_EQ(first[0], second[0]);
  EXPECT_EQ(first[1], second[1]);
}

TEST(DataAssociation, EmptyCostShapesAndMalformedMatrixAreRejected) {
  EXPECT_TRUE(prediction::DataAssociation::solve_cost_matrix(matrix_from(0, 4, {})).empty());
  const std::vector<std::optional<std::size_t>> no_columns =
      prediction::DataAssociation::solve_cost_matrix(matrix_from(3, 0, {}));
  ASSERT_EQ(no_columns.size(), 3U);
  EXPECT_FALSE(no_columns[0].has_value());
  EXPECT_FALSE(no_columns[1].has_value());
  EXPECT_FALSE(no_columns[2].has_value());

  prediction::CostMatrix bad_size = matrix_from(2, 2, {0.0, 1.0});
  EXPECT_THROW(prediction::DataAssociation::solve_cost_matrix(bad_size), std::invalid_argument);
  prediction::CostMatrix negative = matrix_from(1, 1, {-1e-12});
  EXPECT_THROW(prediction::DataAssociation::solve_cost_matrix(negative), std::invalid_argument);
  prediction::CostMatrix non_finite =
      matrix_from(1, 1, {std::numeric_limits<double>::infinity()});
  EXPECT_THROW(prediction::DataAssociation::solve_cost_matrix(non_finite), std::invalid_argument);
}

TEST(DataAssociation, HeaderOnlyAnchorLinksFromAnotherTranslationUnit) {
  const prediction::DataAssociation& anchor = prediction::test_support::data_association_linkage_anchor();
  EXPECT_GT(anchor.config().gate_threshold_sq, 0.0);
  const prediction::DataAssociation local;
  EXPECT_DOUBLE_EQ(anchor.config().gate_threshold_sq, local.config().gate_threshold_sq);
  const prediction::KalmanFilter filter;
  const prediction::AssociationResult result =
      anchor.associate({prediction::TrackState{}}, {prediction::Detection{}}, filter);
  EXPECT_TRUE(result.matches.empty());
  EXPECT_EQ(result.unmatched_tracks.size(), 1U);
  EXPECT_EQ(result.unmatched_detections.size(), 1U);
}
