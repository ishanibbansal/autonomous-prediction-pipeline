#include "prediction/csv_reader.hpp"
#include "prediction/tracker.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace prediction {
namespace test_support {
const MultiObjectTracker& linkage_anchor();
}  // namespace test_support
}  // namespace prediction

namespace {

constexpr double kDt = 0.1;
constexpr double kLength = 4.5;
constexpr double kWidth = 1.9;
constexpr double kPositionTolerance = 1e-6;
constexpr double kExactTolerance = 1e-9;

prediction::Detection make_detection(double x, double y, double heading, double score = 0.9) {
  prediction::Detection detection;
  detection.position = Eigen::Vector2d(x, y);
  detection.size = Eigen::Vector2d(kLength, kWidth);
  detection.heading = heading;
  detection.score = score;
  return detection;
}

prediction::DetectionFrame make_frame(std::uint64_t frame_id, double timestamp,
                                      std::vector<prediction::Detection> detections = {}) {
  prediction::DetectionFrame frame;
  frame.frame_id = frame_id;
  frame.timestamp = timestamp;
  frame.detections = std::move(detections);
  return frame;
}

prediction::DetectionFrame moving_detection(std::uint64_t frame_id, double x0, double y0,
                                            double speed, double heading) {
  const double time = static_cast<double>(frame_id) * kDt;
  return make_frame(frame_id, time,
                    {make_detection(x0 + speed * std::cos(heading) * time,
                                    y0 + speed * std::sin(heading) * time, heading)});
}

void expect_message_contains(const prediction::TrackerConfig& config, const char* field) {
  try {
    prediction::MultiObjectTracker tracker(config);
    FAIL() << "expected std::invalid_argument naming " << field;
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(field), std::string::npos) << error.what();
  }
}

void expect_states_equal(const prediction::TrackState& actual, const prediction::TrackState& expected) {
  EXPECT_EQ(actual.track_id, expected.track_id);
  EXPECT_EQ(actual.status, expected.status);
  EXPECT_DOUBLE_EQ(actual.size.x(), expected.size.x());
  EXPECT_DOUBLE_EQ(actual.size.y(), expected.size.y());
  for (int row = 0; row < actual.state.rows(); ++row) {
    EXPECT_DOUBLE_EQ(actual.state(row), expected.state(row)) << "state " << row;
  }
  for (int row = 0; row < actual.covariance.rows(); ++row) {
    for (int col = 0; col < actual.covariance.cols(); ++col) {
      EXPECT_DOUBLE_EQ(actual.covariance(row, col), expected.covariance(row, col))
          << "covariance (" << row << ", " << col << ")";
    }
  }
}

void expect_records_equal(const prediction::TrackRecord& actual,
                          const prediction::TrackRecord& expected) {
  EXPECT_EQ(actual.hits, expected.hits);
  EXPECT_EQ(actual.consecutive_hits, expected.consecutive_hits);
  EXPECT_EQ(actual.consecutive_misses, expected.consecutive_misses);
  EXPECT_DOUBLE_EQ(actual.birth_time, expected.birth_time);
  EXPECT_DOUBLE_EQ(actual.state_time, expected.state_time);
  EXPECT_DOUBLE_EQ(actual.last_hit_time, expected.last_hit_time);
}

void expect_unchanged(const prediction::MultiObjectTracker& tracker,
                      const std::vector<prediction::TrackState>& tracks,
                      const std::vector<prediction::TrackRecord>& records,
                      const std::vector<prediction::TrackState>& published) {
  ASSERT_EQ(tracker.all_tracks().size(), tracks.size());
  ASSERT_EQ(tracker.records().size(), records.size());
  for (std::size_t index = 0; index < tracks.size(); ++index) {
    expect_states_equal(tracker.all_tracks()[index], tracks[index]);
    expect_records_equal(tracker.records()[index], records[index]);
  }
  const std::vector<prediction::TrackState> after = tracker.tracks(tracker.config().output_policy);
  ASSERT_EQ(after.size(), published.size());
  for (std::size_t index = 0; index < published.size(); ++index) {
    expect_states_equal(after[index], published[index]);
  }
}

double separation(const prediction::TrackState& track, double x, double y) {
  return std::hypot(track.state(0) - x, track.state(1) - y);
}

}  // namespace

TEST(MultiObjectTracker, ConfigValidationNamesTheRejectedField) {
  EXPECT_NO_THROW(prediction::MultiObjectTracker{});

  prediction::TrackerConfig zero_hits;
  zero_hits.min_hits_to_confirm = 0U;
  expect_message_contains(zero_hits, "TrackerConfig::min_hits_to_confirm");

  prediction::TrackerConfig zero_misses;
  zero_misses.max_consecutive_misses = 0U;
  expect_message_contains(zero_misses, "TrackerConfig::max_consecutive_misses");

  prediction::TrackerConfig tentative_exceeds_confirmed;
  tentative_exceeds_confirmed.max_consecutive_misses = 2U;
  tentative_exceeds_confirmed.max_tentative_misses = 3U;
  expect_message_contains(tentative_exceeds_confirmed, "TrackerConfig::max_tentative_misses");

  prediction::TrackerConfig bad_filter;
  bad_filter.filter_config.sigma_pos_sq = 0.0;
  EXPECT_THROW(prediction::MultiObjectTracker{bad_filter}, std::invalid_argument);

  prediction::TrackerConfig bad_gate;
  bad_gate.association_config.gate_threshold_sq = -1.0;
  EXPECT_THROW(prediction::MultiObjectTracker{bad_gate}, std::invalid_argument);
}

TEST(MultiObjectTracker, SequentialIdsFollowDetectionOrderAndReset) {
  prediction::TrackerConfig config;
  config.output_policy = prediction::TrackOutputPolicy::All;
  prediction::MultiObjectTracker tracker(config);

  tracker.process_frame(make_frame(
      1, 0.0,
      {make_detection(0.0, 0.0, 0.0), make_detection(40.0, 0.0, 0.0),
       make_detection(0.0, 40.0, 0.0)}));
  ASSERT_EQ(tracker.all_tracks().size(), 3U);
  EXPECT_EQ(tracker.all_tracks()[0].track_id, 1U);
  EXPECT_EQ(tracker.all_tracks()[1].track_id, 2U);
  EXPECT_EQ(tracker.all_tracks()[2].track_id, 3U);

  tracker.process_frame(make_frame(
      2, kDt,
      {make_detection(0.0, 0.0, 0.0), make_detection(40.0, 0.0, 0.0),
       make_detection(0.0, 40.0, 0.0), make_detection(80.0, 80.0, 0.0)}));
  ASSERT_EQ(tracker.all_tracks().size(), 4U);
  EXPECT_EQ(tracker.all_tracks()[3].track_id, 4U);

  tracker.reset();
  EXPECT_EQ(tracker.active_track_count(), 0U);
  tracker.process_frame(make_frame(0, -2.5, {make_detection(1.0, 2.0, 0.4)}));
  ASSERT_EQ(tracker.all_tracks().size(), 1U);
  EXPECT_EQ(tracker.all_tracks()[0].track_id, 1U);
  EXPECT_EQ(tracker.records()[0].hits, 1U);
}

TEST(MultiObjectTracker, InvalidDetectionsDoNotConsumeIds) {
  const auto run = [](prediction::Detection invalid) {
    prediction::MultiObjectTracker tracker;
    tracker.process_frame(make_frame(
        1, 0.0,
        {make_detection(0.0, 0.0, 0.0), invalid, make_detection(30.0, 5.0, 0.2)}));
    ASSERT_EQ(tracker.active_track_count(), 2U);
    EXPECT_EQ(tracker.all_tracks()[0].track_id, 1U);
    EXPECT_EQ(tracker.all_tracks()[1].track_id, 2U);
    EXPECT_NEAR(tracker.all_tracks()[0].state(0), 0.0, kExactTolerance);
    EXPECT_NEAR(tracker.all_tracks()[1].state(0), 30.0, kExactTolerance);
  };

  prediction::Detection nan_position = make_detection(10.0, 0.0, 0.0);
  nan_position.position.x() = std::numeric_limits<double>::quiet_NaN();
  run(nan_position);

  prediction::Detection zero_size = make_detection(10.0, 0.0, 0.0);
  zero_size.size = Eigen::Vector2d(0.0, kWidth);
  run(zero_size);
}

TEST(MultiObjectTracker, ConfirmsOnTheMthConsecutiveHit) {
  constexpr std::size_t kHitsToConfirm = 3U;
  prediction::TrackerConfig config;
  config.min_hits_to_confirm = kHitsToConfirm;
  config.output_policy = prediction::TrackOutputPolicy::All;
  prediction::MultiObjectTracker tracker(config);

  for (std::uint64_t frame_id = 0; frame_id < kHitsToConfirm; ++frame_id) {
    const double time = static_cast<double>(frame_id) * kDt;
    tracker.process_frame(moving_detection(frame_id, 2.0, -4.0, 8.0, 0.0));
    ASSERT_EQ(tracker.all_tracks().size(), 1U);
    ASSERT_EQ(tracker.records().size(), 1U);
    const prediction::TrackState& track = tracker.all_tracks()[0];
    const prediction::TrackRecord& record = tracker.records()[0];
    EXPECT_EQ(record.hits, static_cast<std::size_t>(frame_id) + 1U);
    EXPECT_EQ(record.consecutive_hits, record.hits);
    EXPECT_EQ(record.consecutive_misses, 0U);
    EXPECT_DOUBLE_EQ(record.birth_time, 0.0);
    EXPECT_DOUBLE_EQ(record.last_hit_time, time);
    EXPECT_DOUBLE_EQ(record.state_time, time);
    if (record.hits < kHitsToConfirm) {
      EXPECT_EQ(track.status, prediction::TrackStatus::Tentative);
    } else {
      EXPECT_EQ(track.status, prediction::TrackStatus::Confirmed);
    }
  }
}

TEST(MultiObjectTracker, OneHitConfirmationIsVisibleAtBirth) {
  prediction::TrackerConfig config;
  config.min_hits_to_confirm = 1U;
  prediction::MultiObjectTracker tracker(config);

  const auto& published =
      tracker.process_frame(make_frame(4, 1.25, {make_detection(3.0, -1.0, 0.5)}));
  ASSERT_EQ(tracker.all_tracks().size(), 1U);
  EXPECT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Confirmed);
  ASSERT_EQ(published.size(), 1U);
  EXPECT_EQ(published[0].track_id, 1U);
  EXPECT_EQ(published[0].status, prediction::TrackStatus::Confirmed);
  EXPECT_EQ(tracker.records()[0].consecutive_hits, 1U);
}

TEST(MultiObjectTracker, SingleFrameFalsePositiveNeverConfirms) {
  prediction::TrackerConfig config;
  config.output_policy = prediction::TrackOutputPolicy::All;
  prediction::MultiObjectTracker tracker(config);

  const auto& first =
      tracker.process_frame(make_frame(1, 0.0, {make_detection(5.0, 5.0, 0.0)}));
  ASSERT_EQ(first.size(), 1U);
  EXPECT_EQ(first[0].status, prediction::TrackStatus::Tentative);
  EXPECT_EQ(tracker.tracks(prediction::TrackOutputPolicy::ConfirmedAndLost).size(), 0U);

  const auto& second = tracker.process_frame(make_frame(2, kDt));
  EXPECT_TRUE(second.empty());
  EXPECT_EQ(tracker.active_track_count(), 0U);
  EXPECT_TRUE(tracker.records().empty());
}

TEST(MultiObjectTracker, TentativeTrackSurvivesOnlyTheConfiguredMisses) {
  prediction::TrackerConfig config;
  config.max_tentative_misses = 1U;
  config.output_policy = prediction::TrackOutputPolicy::All;
  prediction::MultiObjectTracker tracker(config);

  tracker.process_frame(make_frame(1, 0.0, {make_detection(1.0, 1.0, 0.0)}));
  tracker.process_frame(make_frame(2, kDt));
  ASSERT_EQ(tracker.active_track_count(), 1U);
  EXPECT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Tentative);
  EXPECT_EQ(tracker.records()[0].consecutive_hits, 0U);
  EXPECT_EQ(tracker.records()[0].consecutive_misses, 1U);
  EXPECT_EQ(tracker.records()[0].hits, 1U);

  tracker.process_frame(make_frame(3, 2.0 * kDt));
  EXPECT_EQ(tracker.active_track_count(), 0U);
}

TEST(MultiObjectTracker, OcclusionCoastsThenReassociatesWithoutAnIdSwitch) {
  prediction::MultiObjectTracker tracker;
  constexpr std::uint64_t kWarmup = 20;
  constexpr double kSpeed = 12.0;
  for (std::uint64_t frame_id = 0; frame_id < kWarmup; ++frame_id) {
    tracker.process_frame(moving_detection(frame_id, 0.0, 0.0, kSpeed, 0.0));
  }
  ASSERT_EQ(tracker.active_track_count(), 1U);
  const std::uint64_t track_id = tracker.all_tracks()[0].track_id;
  const double x_before = tracker.all_tracks()[0].state(0);
  const double vx = tracker.all_tracks()[0].state(2);
  EXPECT_GT(vx, 0.0);

  const auto& coasted = tracker.process_frame(make_frame(kWarmup, static_cast<double>(kWarmup) * kDt));
  ASSERT_EQ(coasted.size(), 1U);
  EXPECT_EQ(coasted[0].track_id, track_id);
  EXPECT_EQ(coasted[0].status, prediction::TrackStatus::Lost);
  EXPECT_NEAR(coasted[0].state(0), x_before + vx * kDt, kExactTolerance);
  EXPECT_EQ(tracker.tracks(prediction::TrackOutputPolicy::ConfirmedOnly).size(), 0U);

  const double x_mid = tracker.all_tracks()[0].state(0);
  tracker.process_frame(make_frame(kWarmup + 1U, static_cast<double>(kWarmup + 1U) * kDt));
  EXPECT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Lost);
  EXPECT_NEAR(tracker.all_tracks()[0].state(0), x_mid + vx * kDt, kExactTolerance);

  const double return_time = static_cast<double>(kWarmup + 2U) * kDt;
  const auto& recovered = tracker.process_frame(
      make_frame(kWarmup + 2U, return_time, {make_detection(kSpeed * return_time, 0.0, 0.0)}));
  ASSERT_EQ(tracker.active_track_count(), 1U);
  ASSERT_EQ(recovered.size(), 1U);
  EXPECT_EQ(recovered[0].track_id, track_id);
  EXPECT_EQ(recovered[0].status, prediction::TrackStatus::Confirmed);
  EXPECT_EQ(tracker.records()[0].consecutive_misses, 0U);
}

TEST(MultiObjectTracker, CoastUsesOnePredictPerFrame) {
  prediction::MultiObjectTracker tracker;
  constexpr std::uint64_t kWarmup = 15;
  constexpr std::uint64_t kCoastedFrames = 2;
  constexpr double kSpeed = 9.0;
  for (std::uint64_t frame_id = 0; frame_id < kWarmup; ++frame_id) {
    tracker.process_frame(moving_detection(frame_id, 1.0, -2.0, kSpeed, 0.0));
  }
  const double last_hit_time = tracker.records()[0].last_hit_time;
  const double x_last = tracker.all_tracks()[0].state(0);
  const double y_last = tracker.all_tracks()[0].state(1);
  const double vx = tracker.all_tracks()[0].state(2);
  const double vy = tracker.all_tracks()[0].state(3);
  EXPECT_DOUBLE_EQ(last_hit_time, static_cast<double>(kWarmup - 1U) * kDt);

  for (std::uint64_t step = 0; step < kCoastedFrames; ++step) {
    const std::uint64_t frame_id = kWarmup + step;
    const double time = static_cast<double>(frame_id) * kDt;
    tracker.process_frame(make_frame(frame_id, time));
    const double elapsed = static_cast<double>(step + 1U) * kDt;
    EXPECT_NEAR(tracker.all_tracks()[0].state(0), x_last + vx * elapsed, kExactTolerance);
    EXPECT_NEAR(tracker.all_tracks()[0].state(1), y_last + vy * elapsed, kExactTolerance);
    EXPECT_DOUBLE_EQ(tracker.records()[0].state_time, time);
    EXPECT_DOUBLE_EQ(tracker.records()[0].last_hit_time, last_hit_time);
    EXPECT_EQ(tracker.records()[0].consecutive_misses, step + 1U);
  }
}

TEST(MultiObjectTracker, ConfirmedTrackIsPrunedOnTheMissAfterTheLimit) {
  prediction::TrackerConfig config;
  config.max_consecutive_misses = 5U;
  prediction::MultiObjectTracker tracker(config);
  for (std::uint64_t frame_id = 0; frame_id < 3; ++frame_id) {
    tracker.process_frame(moving_detection(frame_id, 0.0, 0.0, 5.0, 0.0));
  }
  ASSERT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Confirmed);

  for (std::uint64_t miss = 1; miss <= 5; ++miss) {
    const std::uint64_t frame_id = 2U + miss;
    tracker.process_frame(make_frame(frame_id, static_cast<double>(frame_id) * kDt));
    ASSERT_EQ(tracker.active_track_count(), 1U) << "miss " << miss;
    EXPECT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Lost);
    EXPECT_EQ(tracker.records()[0].consecutive_misses, miss);
    EXPECT_EQ(tracker.tracks(prediction::TrackOutputPolicy::ConfirmedAndLost).size(), 1U);
  }

  tracker.process_frame(make_frame(8, 8.0 * kDt));
  EXPECT_EQ(tracker.active_track_count(), 0U);
  EXPECT_TRUE(tracker.process_frame(make_frame(9, 9.0 * kDt)).empty());
}

TEST(MultiObjectTracker, RejectedFrameLeavesTracksAndTheNextValidFrameWorks) {
  const auto reject_then_continue = [](const prediction::DetectionFrame& invalid) {
    prediction::MultiObjectTracker tracker;
    const std::vector<prediction::TrackState>* published_ref = nullptr;
    for (std::uint64_t frame_id = 0; frame_id < 3; ++frame_id) {
      published_ref = &tracker.process_frame(moving_detection(frame_id, 0.0, 0.0, 10.0, 0.0));
    }
    ASSERT_NE(published_ref, nullptr);
    const std::vector<prediction::TrackState> tracks = tracker.all_tracks();
    const std::vector<prediction::TrackRecord> records = tracker.records();
    const std::vector<prediction::TrackState> published = *published_ref;
    ASSERT_EQ(tracks[0].status, prediction::TrackStatus::Confirmed);

    EXPECT_THROW(tracker.process_frame(invalid), std::invalid_argument);
    expect_unchanged(tracker, tracks, records, published);
    ASSERT_EQ(published_ref->size(), published.size());
    expect_states_equal((*published_ref)[0], published[0]);

    const auto& continued = tracker.process_frame(moving_detection(3, 0.0, 0.0, 10.0, 0.0));
    ASSERT_EQ(continued.size(), 1U);
    EXPECT_EQ(continued[0].track_id, tracks[0].track_id);
    EXPECT_EQ(continued[0].status, prediction::TrackStatus::Confirmed);
    EXPECT_EQ(tracker.records()[0].hits, records[0].hits + 1U);
    EXPECT_DOUBLE_EQ(tracker.records()[0].state_time, 0.3);
  };

  prediction::DetectionFrame earlier = moving_detection(4, 0.0, 0.0, 10.0, 0.0);
  earlier.timestamp = 0.1;
  reject_then_continue(earlier);

  prediction::DetectionFrame same_id = moving_detection(2, 0.0, 0.0, 10.0, 0.0);
  same_id.timestamp = 0.4;
  reject_then_continue(same_id);

  prediction::DetectionFrame smaller_id = moving_detection(1, 0.0, 0.0, 10.0, 0.0);
  smaller_id.timestamp = 0.4;
  reject_then_continue(smaller_id);

  prediction::DetectionFrame nan_time = moving_detection(4, 0.0, 0.0, 10.0, 0.0);
  nan_time.timestamp = std::numeric_limits<double>::quiet_NaN();
  reject_then_continue(nan_time);

  prediction::DetectionFrame bad_pose = moving_detection(4, 0.0, 0.0, 10.0, 0.0);
  bad_pose.ego_pose_2d.yaw = std::numeric_limits<double>::infinity();
  reject_then_continue(bad_pose);
}

TEST(MultiObjectTracker, FailedPredictPrunesTheTrackImmediately) {
  prediction::TrackerConfig config;
  config.output_policy = prediction::TrackOutputPolicy::All;
  prediction::MultiObjectTracker tracker(config);
  for (std::uint64_t frame_id = 0; frame_id < 3; ++frame_id) {
    tracker.process_frame(moving_detection(frame_id, 0.0, 0.0, 10.0, 0.0));
  }
  ASSERT_EQ(tracker.active_track_count(), 1U);
  ASSERT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Confirmed);
  ASSERT_EQ(tracker.all_tracks()[0].track_id, 1U);

  // dt is about 1e100, so the dt^4 process-noise term overflows to infinity and
  // the filter rejects the predict. The tracker must drop that track at once,
  // not demote it to Lost, and the detection in the same frame is a fresh birth.
  const double huge_time = 1e100;
  const auto& published = tracker.process_frame(
      make_frame(3, huge_time, {make_detection(50.0, 50.0, 0.0)}));
  ASSERT_EQ(tracker.active_track_count(), 1U);
  ASSERT_EQ(published.size(), 1U);
  EXPECT_EQ(published[0].track_id, 2U);
  EXPECT_EQ(published[0].status, prediction::TrackStatus::Tentative);
  EXPECT_EQ(tracker.records()[0].hits, 1U);
  EXPECT_DOUBLE_EQ(tracker.records()[0].birth_time, huge_time);
}

TEST(MultiObjectTracker, EqualTimestampStillCountsAsAHit) {
  prediction::TrackerConfig config;
  config.output_policy = prediction::TrackOutputPolicy::All;
  prediction::MultiObjectTracker tracker(config);

  tracker.process_frame(make_frame(1, 2.0, {make_detection(4.0, -3.0, 0.7)}));
  tracker.process_frame(make_frame(2, 2.0, {make_detection(4.0, -3.0, 0.7)}));
  ASSERT_EQ(tracker.active_track_count(), 1U);
  EXPECT_EQ(tracker.records()[0].hits, 2U);
  EXPECT_EQ(tracker.records()[0].consecutive_hits, 2U);
  EXPECT_EQ(tracker.records()[0].consecutive_misses, 0U);
  EXPECT_DOUBLE_EQ(tracker.records()[0].state_time, 2.0);
  EXPECT_NEAR(tracker.all_tracks()[0].state(0), 4.0, kPositionTolerance);
  EXPECT_NEAR(tracker.all_tracks()[0].state(1), -3.0, kPositionTolerance);
}

TEST(MultiObjectTracker, PoseSourceSelectsOnlyTheConfiguredTransform) {
  prediction::DetectionFrame frame =
      make_frame(1, 0.0, {make_detection(2.0, 0.0, 0.1)});
  frame.ego_pose_2d = {10.0, 5.0, prediction::kPi / 2.0};
  // ego_pose_se3 stays at identity, so the two sources disagree.

  prediction::TrackerConfig planar;
  planar.pose_source = prediction::PoseSource::Planar;
  prediction::MultiObjectTracker planar_tracker(planar);
  planar_tracker.process_frame(frame);
  ASSERT_EQ(planar_tracker.all_tracks().size(), 1U);
  EXPECT_NEAR(planar_tracker.all_tracks()[0].state(0), 10.0, kPositionTolerance);
  EXPECT_NEAR(planar_tracker.all_tracks()[0].state(1), 7.0, kPositionTolerance);
  EXPECT_NEAR(planar_tracker.all_tracks()[0].state(4), prediction::kPi / 2.0 + 0.1,
              kPositionTolerance);

  prediction::DetectionFrame planar_ignores_se3 = frame;
  planar_ignores_se3.frame_id = 2;
  planar_ignores_se3.timestamp = kDt;
  planar_ignores_se3.ego_pose_se3(2, 2) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_NO_THROW(planar_tracker.process_frame(planar_ignores_se3));

  prediction::TrackerConfig se3;
  se3.pose_source = prediction::PoseSource::SE3;
  prediction::MultiObjectTracker se3_tracker(se3);
  se3_tracker.process_frame(frame);
  ASSERT_EQ(se3_tracker.all_tracks().size(), 1U);
  EXPECT_NEAR(se3_tracker.all_tracks()[0].state(0), 2.0, kPositionTolerance);
  EXPECT_NEAR(se3_tracker.all_tracks()[0].state(1), 0.0, kPositionTolerance);
  EXPECT_NEAR(se3_tracker.all_tracks()[0].state(4), 0.1, kPositionTolerance);

  prediction::DetectionFrame bad_se3 = frame;
  bad_se3.frame_id = 2;
  bad_se3.timestamp = kDt;
  bad_se3.ego_pose_se3(0, 3) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(se3_tracker.process_frame(bad_se3), std::invalid_argument);
  EXPECT_EQ(se3_tracker.active_track_count(), 1U);
  EXPECT_NEAR(se3_tracker.all_tracks()[0].state(0), 2.0, kPositionTolerance);

  prediction::DetectionFrame unused_planar_nan = frame;
  unused_planar_nan.frame_id = 2;
  unused_planar_nan.timestamp = kDt;
  unused_planar_nan.ego_pose_2d.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_NO_THROW(se3_tracker.process_frame(unused_planar_nan));
  EXPECT_EQ(se3_tracker.active_track_count(), 1U);
}

TEST(MultiObjectTracker, ShallowCrossingKeepsBothIdentities) {
  // 50 frames, crossing at frame 25. Headings differ by 25 degrees, speeds are
  // 10 and 11 m/s, and both vehicles are at the origin on that frame. Eight
  // hard-coded drops (8% of 100 slots) cover both vehicles; frame 24 is the
  // frame before the intersection and frame 25 is the intersection itself.
  constexpr std::size_t kFrames = 50;
  constexpr std::size_t kIntersection = 25;
  constexpr double kHeading = 25.0 * prediction::kPi / 180.0;
  constexpr double kSpeedA = 10.0;
  constexpr double kSpeedB = 11.0;
  constexpr double kMaxOwnError = 1.0;                                    // meters
  constexpr double kMaxHeadingError = 10.0 * prediction::kPi / 180.0;     // radians
  const double cosine = std::cos(kHeading);
  const double sine = std::sin(kHeading);
  const bool drop_a[] = {false, false, false, false, false, false, false, false, true,  false,
                         false, false, false, false, false, false, true,  false, false, false,
                         false, false, false, false, true,  false, false, false, false, false,
                         false, false, false, false, false, false, false, false, false, false,
                         true,  false, false, false, false, false, false, false, false, false};
  const bool drop_b[] = {false, false, false, false, false, false, false, false, false, false,
                         false, false, true,  false, false, false, false, false, false, false,
                         false, false, false, false, false, true,  false, false, false, false,
                         false, false, false, true,  false, false, false, false, false, false,
                         false, false, false, false, false, false, true,  false, false, false};
  ASSERT_EQ(kIntersection, 25U);
  EXPECT_TRUE(drop_a[kIntersection - 1U]);
  EXPECT_TRUE(drop_b[kIntersection]);

  prediction::MultiObjectTracker tracker;
  std::size_t drops = 0;
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    SCOPED_TRACE(frame);
    const double time = static_cast<double>(frame) * kDt;
    const double along = time - static_cast<double>(kIntersection) * kDt;
    const double ax = kSpeedA * along;
    const double ay = 0.0;
    const double bx = kSpeedB * along * cosine;
    const double by = kSpeedB * along * sine;
    if (drop_a[frame]) {
      ++drops;
    }
    if (drop_b[frame]) {
      ++drops;
    }

    std::vector<prediction::Detection> detections;
    if (!drop_a[frame]) {
      detections.push_back(make_detection(ax, ay, 0.0));
    }
    if (!drop_b[frame]) {
      detections.push_back(make_detection(bx, by, kHeading));
    }
    tracker.process_frame(make_frame(static_cast<std::uint64_t>(frame), time, detections));

    ASSERT_EQ(tracker.active_track_count(), 2U);
    const prediction::TrackState& track_a = tracker.all_tracks()[0];
    const prediction::TrackState& track_b = tracker.all_tracks()[1];
    EXPECT_EQ(track_a.track_id, 1U);
    EXPECT_EQ(track_b.track_id, 2U);
    EXPECT_LE(separation(track_a, ax, ay), separation(track_a, bx, by) + kExactTolerance);
    EXPECT_LE(separation(track_b, bx, by), separation(track_b, ax, ay) + kExactTolerance);
    if (frame >= 2U) {
      EXPECT_NE(track_a.status, prediction::TrackStatus::Tentative);
      EXPECT_NE(track_b.status, prediction::TrackStatus::Tentative);
      // The nearest-vehicle check above is vacuous when both vehicles share a
      // point, and it tolerates a track that drifted away. Bound the absolute
      // error against its own vehicle as well.
      EXPECT_LT(separation(track_a, ax, ay), kMaxOwnError);
      EXPECT_LT(separation(track_b, bx, by), kMaxOwnError);
    }
    // After the paths have separated, each track's velocity must point along its
    // own vehicle's heading, so a swapped identity (which would carry the other
    // vehicle's velocity) cannot pass.
    if (frame >= kIntersection + 5U) {
      const double heading_a = std::atan2(track_a.state(3), track_a.state(2));
      const double heading_b = std::atan2(track_b.state(3), track_b.state(2));
      EXPECT_LT(std::abs(prediction::normalize_angle(heading_a - 0.0)), kMaxHeadingError);
      EXPECT_LT(std::abs(prediction::normalize_angle(heading_b - kHeading)), kMaxHeadingError);
    }
  }
  EXPECT_EQ(drops, 8U);
  EXPECT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Confirmed);
  EXPECT_EQ(tracker.all_tracks()[1].status, prediction::TrackStatus::Confirmed);
}

TEST(MultiObjectTracker, CsvStreamTracksOneWorldObjectThroughAMovingEgo) {
  constexpr std::size_t kFrames = 12;
  constexpr std::size_t kSentinel = 6;
  constexpr double kSpeed = 6.0;
  constexpr double kHeading = 0.2;
  constexpr double kYaw = 0.35;

  std::ostringstream csv;
  csv << "frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,score\n";
  csv << std::setprecision(17);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    const double time = static_cast<double>(frame) * kDt;
    const prediction::Pose2D ego{1.5 * time, -0.4 * time, kYaw};
    csv << frame << ',' << time << ',' << ego.x << ',' << ego.y << ',' << ego.yaw << ',';
    if (frame == kSentinel) {
      csv << ",,,,,\n";
      continue;
    }
    prediction::Detection world = make_detection(8.0 + kSpeed * std::cos(kHeading) * time,
                                                -3.0 + kSpeed * std::sin(kHeading) * time,
                                                kHeading, 0.8);
    const prediction::Detection ego_detection = prediction::transform_world_to_ego(world, ego);
    csv << ego_detection.position.x() << ',' << ego_detection.position.y() << ','
        << ego_detection.size.x() << ',' << ego_detection.size.y() << ','
        << ego_detection.heading << ',' << ego_detection.score << '\n';
  }

  std::istringstream input(csv.str());
  prediction::CsvReader reader(input);
  prediction::MultiObjectTracker tracker;
  prediction::DetectionFrame frame;
  std::size_t frames_read = 0;
  while (reader.read_next_frame(frame)) {
    SCOPED_TRACE(frames_read);
    const auto& published = tracker.process_frame(frame);
    ASSERT_EQ(tracker.active_track_count(), 1U);
    const double time = static_cast<double>(frames_read) * kDt;
    const double truth_x = 8.0 + kSpeed * std::cos(kHeading) * time;
    const double truth_y = -3.0 + kSpeed * std::sin(kHeading) * time;
    if (frames_read == kSentinel) {
      EXPECT_TRUE(frame.detections.empty());
      EXPECT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Lost);
      ASSERT_EQ(published.size(), 1U);
      EXPECT_EQ(published[0].status, prediction::TrackStatus::Lost);
    } else if (frames_read >= 2U) {
      EXPECT_EQ(tracker.all_tracks()[0].status, prediction::TrackStatus::Confirmed);
    }
    if (frames_read >= 2U) {
      EXPECT_LT(separation(tracker.all_tracks()[0], truth_x, truth_y), 0.5);
    }
    ++frames_read;
  }
  EXPECT_EQ(frames_read, kFrames);
  EXPECT_EQ(tracker.all_tracks()[0].track_id, 1U);
  EXPECT_EQ(tracker.records()[0].hits, kFrames - 1U);
}

TEST(MultiObjectTracker, HeaderOnlyAnchorLinksFromAnotherTranslationUnit) {
  const prediction::MultiObjectTracker& anchor = prediction::test_support::linkage_anchor();
  EXPECT_EQ(anchor.config().min_hits_to_confirm, 3U);
  EXPECT_EQ(anchor.active_track_count(), 0U);
  const prediction::MultiObjectTracker local;
  EXPECT_EQ(anchor.config().max_consecutive_misses, local.config().max_consecutive_misses);
  EXPECT_EQ(anchor.config().output_policy, local.config().output_policy);
}
