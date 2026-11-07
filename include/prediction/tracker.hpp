#pragma once

#include "prediction/data_association.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prediction {

/// Which live tracks process_frame and tracks() copy into their result.
enum class TrackOutputPolicy { ConfirmedOnly, ConfirmedAndLost, All };

/// Which DetectionFrame pose is used to lift ego-frame detections into the world.
/// The other pose is ignored, including when it is non-finite.
enum class PoseSource { Planar, SE3 };

/// Lifecycle and output settings for MultiObjectTracker.
///
/// This struct is a value. It is not validated until it is copied into a
/// MultiObjectTracker. A caller may build an invalid config and repair it before
/// construction; the tracker then keeps its own validated copy. The nested filter
/// and association configs are validated by those types' constructors.
struct TrackerConfig {
  KalmanFilterConfig filter_config{};
  DataAssociationConfig association_config{};
  std::size_t min_hits_to_confirm{3U};     // >= 1; consecutive hits, including birth
  std::size_t max_consecutive_misses{5U};  // >= 1; prune when misses exceed this
  std::size_t max_tentative_misses{0U};    // <= max_consecutive_misses; 0 drops a Tentative on any miss
  TrackOutputPolicy output_policy{TrackOutputPolicy::ConfirmedAndLost};
  PoseSource pose_source{PoseSource::Planar};
};

/// Per-track bookkeeping kept in lockstep with MultiObjectTracker's TrackState vector.
/// TrackState itself has no counters or timestamps, and this record is not copied
/// into the association input.
struct TrackRecord {
  std::size_t hits{0U};                // lifetime matched updates, including birth
  std::size_t consecutive_hits{0U};
  std::size_t consecutive_misses{0U};
  double birth_time{0.0};
  double state_time{0.0};              // time at which the parallel TrackState is valid
  double last_hit_time{0.0};
};

/// Multi-object tracker: predict, associate, update, coast, confirm, and prune.
///
/// Live tracks are always stored in ascending track_id order. IDs start at 1 and
/// increase by one for each successful birth. ID 0 is never assigned, matching
/// TrackState{}. A pruned ID is not recycled. reset() restarts the counter at 1.
///
/// Each accepted frame predicts every live track once, by that frame's dt. A
/// coasting track is not predicted again from last_hit_time. The process-noise
/// model is not invariant to splitting a step, so a second predict would also
/// change the covariance. Every track returned by process_frame or tracks() is
/// therefore valid at the timestamp of the latest accepted frame, including a
/// newborn whose state is the birth detection at that same timestamp.
///
/// The reference returned by process_frame is invalidated by the next
/// process_frame or reset call. tracks() returns a copy and may allocate.
///
/// Frame validation is transactional and has the strong guarantee: a rejected
/// frame throws std::invalid_argument and leaves tracks, records, IDs, history,
/// and the output buffer untouched. After validation, the only exceptions are
/// std::bad_alloc and the assignment solver's "should never happen"
/// std::logic_error. Those have the basic guarantee: the tracker may already
/// have predicted, updated, pruned, or spawned before the throw.
///
/// Once the tracker has warmed up, process_frame reuses member buffers. It does
/// not allocate except inside DataAssociation::associate, which still builds its
/// cost matrix and scratch storage on every call.
class MultiObjectTracker {
 public:
  static constexpr std::size_t kInitialCapacity = 64;

  explicit MultiObjectTracker(TrackerConfig config = {})
      : config_(validated(std::move(config))),
        filter_(config_.filter_config),
        associator_(config_.association_config) {
    tracks_.reserve(kInitialCapacity);
    records_.reserve(kInitialCapacity);
    world_detections_.reserve(kInitialCapacity);
    prune_flags_.reserve(kInitialCapacity);
    output_.reserve(kInitialCapacity);
  }

  /// Run one frame. See the class comment for the lifetime of the returned
  /// reference and for the exception guarantees.
  const std::vector<TrackState>& process_frame(const DetectionFrame& frame) {
    // 1. Validate before any member is written.
    validate_frame(frame);

    // 2. Transform, preserving detection order. Invalid detections stay in the
    // buffer so later indices still match the input; association and birth skip them.
    world_detections_.clear();
    for (const Detection& detection : frame.detections) {
      world_detections_.push_back(transform_detection(detection, frame));
    }

    // 3. One predict per live track, by this frame's dt. A failed predict cannot
    // be associated, so that track is removed before association indices are used.
    const double dt = has_last_frame_ ? frame.timestamp - last_timestamp_ : 0.0;
    prepare_prune_flags();
    for (std::size_t index = 0; index < tracks_.size(); ++index) {
      if (!filter_.predict(tracks_[index], dt)) {
        prune_flags_[index] = static_cast<char>(1);
        continue;
      }
      records_[index].state_time = frame.timestamp;
    }
    compact_pruned();

    // 4. Associate the predicted, still-admissible tracks. tracks_ is the
    // associator's input; it is not copied.
    const AssociationResult result =
        associator_.associate(tracks_, world_detections_, filter_);

    // 5-6. Hits and misses share one prune pass so a failed update is a miss.
    prepare_prune_flags();
    for (const TrackDetectionMatch& match : result.matches) {
      if (!filter_.update(tracks_[match.track_index],
                          world_detections_[match.detection_index])) {
        record_miss(match.track_index);
        continue;
      }
      record_hit(match.track_index, frame.timestamp);
    }
    for (const std::size_t track_index : result.unmatched_tracks) {
      record_miss(track_index);
    }

    // 7. Prune before birth. A newborn is never marked, so this does not change
    // which tracks survive the frame, and the match indices above stay valid.
    compact_pruned();

    // 8. Births in ascending detection-index order. A failed initialize does not
    // consume an ID. Promotion runs immediately so min_hits_to_confirm == 1
    // confirms the track in its birth frame.
    //
    // Reserve both vectors before the first push_back so std::bad_alloc cannot
    // leave tracks_ and records_ with different sizes. Once capacity exists, the
    // push_backs below cannot throw: TrackState and TrackRecord are fixed-size and
    // copying them does not allocate. If the second reserve throws, both sizes are
    // still unchanged. In steady state both reserves are no-ops.
    const std::size_t birth_capacity = tracks_.size() + result.unmatched_detections.size();
    tracks_.reserve(birth_capacity);
    records_.reserve(birth_capacity);
    for (const std::size_t detection_index : result.unmatched_detections) {
      const std::optional<TrackState> born = filter_.initialize(
          world_detections_[detection_index], next_track_id_, TrackStatus::Tentative);
      if (!born.has_value()) {
        continue;
      }
      TrackState track = *born;
      TrackRecord record;
      record.hits = 1U;
      record.consecutive_hits = 1U;
      record.consecutive_misses = 0U;
      record.birth_time = frame.timestamp;
      record.state_time = frame.timestamp;
      record.last_hit_time = frame.timestamp;
      promote_after_hit(track, record);
      tracks_.push_back(std::move(track));
      records_.push_back(record);
      ++next_track_id_;
    }

    // 9. History is committed only after the frame's track edits have been applied.
    last_frame_id_ = frame.frame_id;
    last_timestamp_ = frame.timestamp;
    has_last_frame_ = true;

    // 10. Publish the policy view. Clear keeps the buffer's capacity.
    publish(output_, config_.output_policy);
    return output_;
  }

  /// Copy of the live tracks that match policy, in ascending track_id order.
  /// May allocate. States are those of the latest accepted frame.
  std::vector<TrackState> tracks(TrackOutputPolicy policy) const {
    std::vector<TrackState> selected;
    publish(selected, policy);
    return selected;
  }

  /// Every live track, every status, ascending track_id. Parallel to records().
  const std::vector<TrackState>& all_tracks() const { return tracks_; }

  /// Bookkeeping parallel to all_tracks(): records()[i] belongs to all_tracks()[i].
  const std::vector<TrackRecord>& records() const { return records_; }

  /// Count of every live track, including Tentative and Lost.
  std::size_t active_track_count() const { return tracks_.size(); }

  const TrackerConfig& config() const { return config_; }

  /// Drop every live track and forget frame history. Buffer capacity is kept.
  /// The next frame may have any frame_id and any finite timestamp. The next
  /// birth is track ID 1 again.
  void reset() {
    tracks_.clear();
    records_.clear();
    output_.clear();
    world_detections_.clear();
    prune_flags_.clear();
    next_track_id_ = 1U;
    last_frame_id_ = 0U;
    last_timestamp_ = 0.0;
    has_last_frame_ = false;
  }

 private:
  static TrackerConfig validated(TrackerConfig config) {
    if (config.min_hits_to_confirm == 0U) {
      throw std::invalid_argument("TrackerConfig::min_hits_to_confirm must be >= 1");
    }
    if (config.max_consecutive_misses == 0U) {
      throw std::invalid_argument("TrackerConfig::max_consecutive_misses must be >= 1");
    }
    if (config.max_tentative_misses > config.max_consecutive_misses) {
      throw std::invalid_argument(
          "TrackerConfig::max_tentative_misses must be <= max_consecutive_misses");
    }
    return config;
  }

  void validate_frame(const DetectionFrame& frame) const {
    if (!std::isfinite(frame.timestamp)) {
      throw std::invalid_argument("DetectionFrame::timestamp must be finite");
    }
    if (config_.pose_source == PoseSource::Planar) {
      const Pose2D& pose = frame.ego_pose_2d;
      if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.yaw)) {
        throw std::invalid_argument("DetectionFrame::ego_pose_2d must be finite");
      }
    } else if (!frame.ego_pose_se3.allFinite()) {
      throw std::invalid_argument("DetectionFrame::ego_pose_se3 must be finite");
    }
    if (!has_last_frame_) {
      return;
    }
    if (!(frame.frame_id > last_frame_id_)) {
      throw std::invalid_argument("DetectionFrame::frame_id must be strictly increasing");
    }
    if (frame.timestamp < last_timestamp_) {
      throw std::invalid_argument("DetectionFrame::timestamp must be non-decreasing");
    }
  }

  Detection transform_detection(const Detection& detection, const DetectionFrame& frame) const {
    if (config_.pose_source == PoseSource::Planar) {
      return transform_ego_to_world(detection, frame.ego_pose_2d);
    }
    return transform_ego_to_world(detection, frame.ego_pose_se3);
  }

  void prepare_prune_flags() {
    prune_flags_.assign(tracks_.size(), static_cast<char>(0));
  }

  /// Compact tracks_ and records_ together, preserving relative order.
  void compact_pruned() {
    std::size_t write = 0U;
    for (std::size_t read = 0U; read < tracks_.size(); ++read) {
      if (prune_flags_[read] != 0) {
        continue;
      }
      if (write != read) {
        tracks_[write] = tracks_[read];
        records_[write] = records_[read];
      }
      ++write;
    }
    tracks_.resize(write);
    records_.resize(write);
  }

  void record_hit(std::size_t index, double timestamp) {
    TrackRecord& record = records_[index];
    record.state_time = timestamp;
    record.last_hit_time = timestamp;
    record.hits += 1U;
    record.consecutive_hits += 1U;
    record.consecutive_misses = 0U;
    promote_after_hit(tracks_[index], record);
  }

  void promote_after_hit(TrackState& track, const TrackRecord& record) const {
    if (track.status == TrackStatus::Lost) {
      track.status = TrackStatus::Confirmed;
      return;
    }
    if (track.status == TrackStatus::Tentative &&
        record.consecutive_hits >= config_.min_hits_to_confirm) {
      track.status = TrackStatus::Confirmed;
    }
  }

  void record_miss(std::size_t index) {
    TrackState& track = tracks_[index];
    TrackRecord& record = records_[index];
    record.consecutive_hits = 0U;
    record.consecutive_misses += 1U;
    if (track.status == TrackStatus::Confirmed) {
      track.status = TrackStatus::Lost;
    }
    const bool tentative_expired = track.status == TrackStatus::Tentative &&
                                   record.consecutive_misses > config_.max_tentative_misses;
    const bool stale = record.consecutive_misses > config_.max_consecutive_misses;
    if (tentative_expired || stale) {
      prune_flags_[index] = static_cast<char>(1);
    }
  }

  void publish(std::vector<TrackState>& destination, TrackOutputPolicy policy) const {
    destination.clear();
    for (const TrackState& track : tracks_) {
      if (included(track.status, policy)) {
        destination.push_back(track);
      }
    }
  }

  static bool included(TrackStatus status, TrackOutputPolicy policy) {
    switch (policy) {
      case TrackOutputPolicy::ConfirmedOnly:
        return status == TrackStatus::Confirmed;
      case TrackOutputPolicy::ConfirmedAndLost:
        return status == TrackStatus::Confirmed || status == TrackStatus::Lost;
      case TrackOutputPolicy::All:
        return true;
    }
    return false;
  }

  TrackerConfig config_;
  KalmanFilter filter_;
  DataAssociation associator_;
  std::vector<TrackState> tracks_;           // passed directly to associate(); never copied
  std::vector<TrackRecord> records_;         // records_[i] belongs to tracks_[i]
  std::vector<Detection> world_detections_;  // reused scratch buffer
  std::vector<char> prune_flags_;            // reused scratch buffer
  std::vector<TrackState> output_;           // what process_frame returns a reference to
  std::uint64_t next_track_id_{1U};
  std::uint64_t last_frame_id_{0U};
  double last_timestamp_{0.0};
  bool has_last_frame_{false};
};

}  // namespace prediction
