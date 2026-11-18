#pragma once

#include "prediction/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prediction {

/// Thresholds for unweighted geometric trajectory scoring.
///
/// This struct is a value. It is not validated until it is copied into a
/// TrajectoryEvaluator. A caller may build an invalid config and repair it
/// before construction; the evaluator then keeps its own validated copy.
///
/// miss_threshold_meters is the inclusive final-displacement hit bound, in
/// meters. timestamp_tolerance is an inclusive absolute timestamp bound, in
/// seconds, not a relative tolerance. Zero tolerance is exact equality.
struct MetricsConfig {
  double miss_threshold_meters{2.0};
  double timestamp_tolerance{1e-4};
};

/// TrajectoryMetrics lives in types.hpp so the visualizer can consume scores
/// without including this header. Callers of metrics.hpp still see the type.

/// Equal-weight summary of a positional prediction/ground-truth batch.
///
/// evaluated_count is the number of pairs. mean_min_ade and mean_min_fde
/// average the per-pair minima, one weight per pair, not per waypoint.
/// miss_rate is the integer miss count divided by the pair count. per_track
/// preserves input order.
struct BatchMetricsSummary {
  std::size_t evaluated_count{0};
  double mean_min_ade{0.0};
  double mean_min_fde{0.0};
  double miss_rate{0.0};
  std::vector<TrajectoryMetrics> per_track{};
};

/// Header-only minADE / minFDE / miss-rate evaluator.
///
/// Metrics are unweighted Euclidean geometry. Probability, velocity, heading,
/// covariance, current_state, and the legacy PredictedTrajectory::waypoints
/// view are never read, including for validation. Every supplied mode is
/// scored in input order. The evaluator does not assume three modes, drop
/// modes, or fall back to the legacy view.
///
/// Ground truth is validated once. Used position components and timestamps
/// must be finite, and timestamps must strictly increase within the ground
/// truth and, separately, within each predicted mode. A single point is valid.
/// Timestamps need not be positive, uniformly spaced, or related to an inferred
/// current time. Each mode must contain exactly the ground-truth waypoint count.
/// At each index the absolute timestamp difference must be at most
/// timestamp_tolerance. Samples are not sorted, interpolated, truncated, or
/// resampled. Finite timestamps whose difference is not a finite double cannot
/// match any allowed tolerance and are rejected as misalignment.
///
/// For each mode, ADE is the arithmetic mean of std::hypot(dx, dy) in waypoint
/// order. FDE is the distance at the last index. This is not RMS and not
/// time-weighted. Running sums are checked doubles. A non-finite coordinate
/// difference, distance, or displacement sum throws std::overflow_error, even
/// in a mode that would not win and even when the mathematical mean would fit.
/// Sums are not replaced by an online mean, and the implementation does not
/// rely on long double having greater range. Small representable distances are
/// not errors. Invalid geometry in a losing mode invalidates the pair; checking
/// does not stop after a perfect mode.
///
/// Construction throws std::invalid_argument naming the offending config field.
/// evaluate and evaluate_batch throw std::invalid_argument for invalid structure,
/// IDs, used fields, time order, or alignment, with track, mode, and waypoint
/// context where applicable. No precedence is required between independently
/// invalid fields. Overflow of a checked sum throws std::overflow_error.
///
/// evaluate_batch scores prediction i against ground truth i. It does not join
/// by ID, and duplicate or unsorted IDs are preserved. Different pairs may have
/// different horizon and mode counts. An empty or unequal-length batch, any
/// invalid pair, or an aggregate overflow rejects the whole call. Inputs are
/// not mutated, and no partial summary is returned.
///
/// Scoring is linear in the number of samples inspected and uses constant
/// auxiliary scoring space per pair, apart from the returned batch storage.
/// The definitions are header-only and safe to include from multiple translation
/// units.
class TrajectoryEvaluator {
 public:
  explicit TrajectoryEvaluator(MetricsConfig config = {}) : config_(validate(std::move(config))) {}

  const MetricsConfig& config() const { return config_; }

  /// Score one prediction against the ground truth of the same track id.
  TrajectoryMetrics evaluate(const PredictedTrajectory& prediction,
                             const GroundTruthTrajectory& ground_truth) const {
    return score_pair(prediction, ground_truth, nullptr);
  }

  /// Score equal-length batches positionally. One invalid pair rejects the call.
  BatchMetricsSummary evaluate_batch(const std::vector<PredictedTrajectory>& predictions,
                                      const std::vector<GroundTruthTrajectory>& ground_truths) const {
    if (predictions.empty() || ground_truths.empty() || predictions.size() != ground_truths.size()) {
      throw std::invalid_argument(
          "TrajectoryEvaluator::evaluate_batch requires nonempty equal-length batches");
    }

    BatchMetricsSummary summary;
    summary.per_track.reserve(predictions.size());
    double ade_sum = 0.0;
    double fde_sum = 0.0;
    std::size_t miss_count = 0;
    for (std::size_t index = 0; index < predictions.size(); ++index) {
      summary.per_track.push_back(score_pair(predictions[index], ground_truths[index], &index));
      ade_sum = add_checked(ade_sum, summary.per_track.back().min_ade, batch_overflow(index));
      fde_sum = add_checked(fde_sum, summary.per_track.back().min_fde, batch_overflow(index));
      if (summary.per_track.back().is_miss) {
        ++miss_count;
      }
    }

    const double pair_count = static_cast<double>(predictions.size());
    summary.evaluated_count = predictions.size();
    summary.mean_min_ade = ade_sum / pair_count;
    summary.mean_min_fde = fde_sum / pair_count;
    summary.miss_rate = static_cast<double>(miss_count) / pair_count;
    if (!std::isfinite(summary.mean_min_ade) || !std::isfinite(summary.mean_min_fde) ||
        !std::isfinite(summary.miss_rate)) {
      throw std::overflow_error(
          "TrajectoryEvaluator batch aggregate is non-finite after scoring every pair");
    }
    return summary;
  }

 private:
  static MetricsConfig validate(MetricsConfig config) {
    if (!std::isfinite(config.miss_threshold_meters) || !(config.miss_threshold_meters > 0.0)) {
      throw std::invalid_argument(
          "MetricsConfig::miss_threshold_meters must be finite and strictly positive");
    }
    if (!std::isfinite(config.timestamp_tolerance) || config.timestamp_tolerance < 0.0) {
      throw std::invalid_argument(
          "MetricsConfig::timestamp_tolerance must be finite and nonnegative");
    }
    return config;
  }

  static std::string track_label(std::uint64_t track_id, const std::size_t* batch_index) {
    std::string label = "track_id " + std::to_string(track_id);
    if (batch_index != nullptr) {
      label += ", batch index " + std::to_string(*batch_index);
    }
    return label;
  }

  static std::string batch_overflow(std::size_t index) {
    return "batch index " + std::to_string(index);
  }

  static void require_finite(double value, const std::string& message) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(message);
    }
  }

  /// Inclusive absolute timestamp match. A non-finite difference is a mismatch:
  /// opposite-sign extremes cannot satisfy any finite nonnegative tolerance.
  static bool timestamps_aligned(double predicted, double truth, double tolerance) {
    const double difference = predicted - truth;
    if (!std::isfinite(difference)) {
      return false;
    }
    return std::abs(difference) <= tolerance;
  }

  static double add_checked(double sum, double term, const std::string& where) {
    if (!std::isfinite(term)) {
      throw std::overflow_error("non-finite displacement (" + where + ")");
    }
    const double next = sum + term;
    if (!std::isfinite(next)) {
      throw std::overflow_error("displacement sum overflow (" + where + ")");
    }
    return next;
  }

  static double distance_checked(const Eigen::Vector2d& predicted, const Eigen::Vector2d& truth,
                                 const std::string& where) {
    const double dx = predicted.x() - truth.x();
    const double dy = predicted.y() - truth.y();
    if (!std::isfinite(dx) || !std::isfinite(dy)) {
      throw std::overflow_error("coordinate difference overflow (" + where + ")");
    }
    const double distance = std::hypot(dx, dy);
    if (!std::isfinite(distance)) {
      throw std::overflow_error("displacement norm overflow (" + where + ")");
    }
    return distance;
  }

  void validate_ground_truth(const GroundTruthTrajectory& ground_truth,
                             const std::size_t* batch_index) const {
    const std::string where = track_label(ground_truth.track_id, batch_index);
    if (ground_truth.waypoints.empty()) {
      throw std::invalid_argument("GroundTruthTrajectory::waypoints must be nonempty (" + where + ")");
    }
    double previous_timestamp = 0.0;
    for (std::size_t index = 0; index < ground_truth.waypoints.size(); ++index) {
      const GroundTruthWaypoint& waypoint = ground_truth.waypoints[index];
      const std::string sample =
          where + ", ground-truth waypoint " + std::to_string(index);
      require_finite(waypoint.timestamp,
                     "GroundTruthWaypoint::timestamp must be finite (" + sample + ")");
      require_finite(waypoint.position.x(),
                     "GroundTruthWaypoint::position.x must be finite (" + sample + ")");
      require_finite(waypoint.position.y(),
                     "GroundTruthWaypoint::position.y must be finite (" + sample + ")");
      if (index > 0 && !(waypoint.timestamp > previous_timestamp)) {
        throw std::invalid_argument(
            "GroundTruthTrajectory timestamps must strictly increase (" + sample + ")");
      }
      previous_timestamp = waypoint.timestamp;
    }
  }

  void validate_mode_structure(const PredictedTrajectory& prediction, std::size_t mode_index,
                               std::size_t expected_count, const std::size_t* batch_index) const {
    const std::string where = track_label(prediction.track_id, batch_index) + ", mode " +
                              std::to_string(mode_index);
    const TrajectoryMode& mode = prediction.modes[mode_index];
    if (mode.waypoints.size() != expected_count) {
      throw std::invalid_argument(
          "TrajectoryMode::waypoints size must equal ground-truth waypoint count (" + where + ")");
    }

    double previous_timestamp = 0.0;
    for (std::size_t index = 0; index < mode.waypoints.size(); ++index) {
      const TrajectoryPoint& point = mode.waypoints[index];
      const std::string sample = where + ", waypoint " + std::to_string(index);
      require_finite(point.timestamp,
                     "TrajectoryPoint::timestamp must be finite (" + sample + ")");
      require_finite(point.position.x(),
                     "TrajectoryPoint::position.x must be finite (" + sample + ")");
      require_finite(point.position.y(),
                     "TrajectoryPoint::position.y must be finite (" + sample + ")");
      if (index > 0 && !(point.timestamp > previous_timestamp)) {
        throw std::invalid_argument(
            "TrajectoryMode timestamps must strictly increase (" + sample + ")");
      }
      previous_timestamp = point.timestamp;
    }
  }

  void validate_alignment(const PredictedTrajectory& prediction, std::size_t mode_index,
                          const GroundTruthTrajectory& ground_truth,
                          const std::size_t* batch_index) const {
    const TrajectoryMode& mode = prediction.modes[mode_index];
    for (std::size_t index = 0; index < mode.waypoints.size(); ++index) {
      if (timestamps_aligned(mode.waypoints[index].timestamp, ground_truth.waypoints[index].timestamp,
                             config_.timestamp_tolerance)) {
        continue;
      }
      throw std::invalid_argument(
          "predicted and ground-truth timestamps differ by more than timestamp_tolerance (" +
          track_label(prediction.track_id, batch_index) + ", mode " + std::to_string(mode_index) +
          ", waypoint " + std::to_string(index) + ")");
    }
  }

  struct ModeScore {
    double ade{0.0};
    double fde{0.0};
  };

  ModeScore score_mode(const PredictedTrajectory& prediction, std::size_t mode_index,
                       const GroundTruthTrajectory& ground_truth,
                       const std::size_t* batch_index) const {
    const TrajectoryMode& mode = prediction.modes[mode_index];
    const std::size_t count = mode.waypoints.size();
    double distance_sum = 0.0;
    double final_distance = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
      const std::string where = track_label(prediction.track_id, batch_index) + ", mode " +
                                std::to_string(mode_index) + ", waypoint " + std::to_string(index);
      const double distance =
          distance_checked(mode.waypoints[index].position, ground_truth.waypoints[index].position, where);
      distance_sum = add_checked(distance_sum, distance, where);
      final_distance = distance;
    }
    const double mean = distance_sum / static_cast<double>(count);
    if (!std::isfinite(mean)) {
      throw std::overflow_error(
          "mode ADE is non-finite (" + track_label(prediction.track_id, batch_index) + ", mode " +
          std::to_string(mode_index) + ")");
    }
    ModeScore score;
    score.ade = mean;
    score.fde = final_distance;
    return score;
  }

  TrajectoryMetrics score_pair(const PredictedTrajectory& prediction,
                               const GroundTruthTrajectory& ground_truth,
                               const std::size_t* batch_index) const {
    if (prediction.track_id != ground_truth.track_id) {
      throw std::invalid_argument(
          "PredictedTrajectory::track_id must equal GroundTruthTrajectory::track_id (" +
          track_label(prediction.track_id, batch_index) + ", ground-truth track_id " +
          std::to_string(ground_truth.track_id) + ")");
    }
    if (prediction.modes.empty()) {
      throw std::invalid_argument(
          "PredictedTrajectory::modes must be nonempty (" +
          track_label(prediction.track_id, batch_index) + ")");
    }
    validate_ground_truth(ground_truth, batch_index);

    const std::size_t waypoint_count = ground_truth.waypoints.size();
    for (std::size_t mode_index = 0; mode_index < prediction.modes.size(); ++mode_index) {
      validate_mode_structure(prediction, mode_index, waypoint_count, batch_index);
      validate_alignment(prediction, mode_index, ground_truth, batch_index);
    }

    TrajectoryMetrics metrics;
    metrics.track_id = prediction.track_id;
    bool seeded = false;
    for (std::size_t mode_index = 0; mode_index < prediction.modes.size(); ++mode_index) {
      const ModeScore score = score_mode(prediction, mode_index, ground_truth, batch_index);
      if (!seeded || score.ade < metrics.min_ade) {
        metrics.min_ade = score.ade;
        metrics.best_ade_mode = mode_index;
      }
      if (!seeded || score.fde < metrics.min_fde) {
        metrics.min_fde = score.fde;
        metrics.best_fde_mode = mode_index;
      }
      seeded = true;
    }
    metrics.is_miss = metrics.min_fde > config_.miss_threshold_meters;
    return metrics;
  }

  MetricsConfig config_;
};

}  // namespace prediction
