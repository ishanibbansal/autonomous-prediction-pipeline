#pragma once

#include "prediction/kalman_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prediction {

/// Gate used by DataAssociation. This struct is a value and is not checked until
/// it is copied into DataAssociation. A caller may repair an invalid config
/// before construction; the associator then keeps its own validated copy.
///
/// gate_threshold_sq is the squared-Mahalanobis acceptance bound, not a
/// Mahalanobis distance. The default is chi^2 with 3 degrees of freedom at
/// probability 0.99, the usual bound for the world-frame observation [x, y, psi].
struct DataAssociationConfig {
  double gate_threshold_sq{11.344866730144373};
};

/// One globally selected track/detection pair.
///
/// mahalanobis_distance_sq is the original squared distance returned by the
/// filter, not the normalized cost passed to the assignment solver.
struct TrackDetectionMatch {
  std::size_t track_index{0};
  std::size_t detection_index{0};
  double mahalanobis_distance_sq{0.0};
};

/// A partition of every supplied track index and every supplied detection index.
/// Indices that fail the filter's admissibility checks, or that have no
/// gate-admissible partner left after the global assignment, are unmatched.
struct AssociationResult {
  std::vector<TrackDetectionMatch> matches;
  std::vector<std::size_t> unmatched_tracks;
  std::vector<std::size_t> unmatched_detections;
};

/// Dense rectangular cost matrix for the low-level assignment seam.
///
/// Dimensions are stored separately so an empty matrix still records whether it
/// is 0 x N or M x 0. costs is row-major and has track_count * detection_count
/// entries. A missing optional is a forbidden edge, not an infinite cost.
struct CostMatrix {
  std::size_t track_count{0};
  std::size_t detection_count{0};
  std::vector<std::optional<double>> costs;
};

/// Gated global association of predicted tracks to world-frame detections.
///
/// associate does not predict, transform, or update. Tracks are used as supplied,
/// and every Detection is assumed to already be in the same world frame as the
/// track state. Detection does not record its frame, so that assumption cannot
/// be checked here.
///
/// A pair is admissible only when KalmanFilter::mahalanobis_distance_squared
/// returns a value and that value is less than or equal to gate_threshold_sq.
/// The comparison has no extra tolerance. nullopt — including a malformed
/// detection, a non-admissible track, or a non-positive-definite innovation
/// covariance — forbids that one edge. Other edges are unaffected.
///
/// Among admissible one-to-one matchings, the result has maximum cardinality.
/// Among those, it minimizes the sum of squared Mahalanobis distances. Equal
/// objectives follow the deterministic shortest-augmenting-path tie break below.
/// Matches are ordered by ascending track index. Both unmatched vectors are
/// ordered by ascending index.
///
/// The assignment is a rectangular Jonker-Volgenant / Kuhn-Munkres reduction.
/// Admissible costs are divided by G = gate_threshold_sq, so each real edge
/// costs at most 1. Each track also has a dummy non-assignment column whose
/// cost is P = min(M, N) + 1. A matching with k real pairs and M - k dummies
/// then has total cost in [(M - k) P, (M - k) P + min(M, N)]. Those intervals
/// do not overlap, so adding one match always beats every possible reduction in
/// the sum of real costs. Forbidden edges are omitted from the augmenting-path
/// search; they are not encoded as large finite costs.
class DataAssociation {
 public:
  explicit DataAssociation(DataAssociationConfig config = {}) : config_(std::move(config)) {
    if (!std::isfinite(config_.gate_threshold_sq) || !(config_.gate_threshold_sq > 0.0)) {
      throw std::invalid_argument(
          "DataAssociationConfig::gate_threshold_sq must be finite and strictly positive");
    }
  }

  const DataAssociationConfig& config() const { return config_; }

  /// Associate tracks[i] with detections_world[j]. Neither input is modified.
  AssociationResult associate(const std::vector<TrackState>& tracks,
                              const std::vector<Detection>& detections_world,
                              const KalmanFilter& filter) const {
    const std::size_t track_count = tracks.size();
    const std::size_t detection_count = detections_world.size();
    CostMatrix costs;
    costs.track_count = track_count;
    costs.detection_count = detection_count;
    costs.costs.resize(track_count * detection_count);

    std::vector<double> accepted_distances(track_count * detection_count, 0.0);
    const double gate = config_.gate_threshold_sq;
    for (std::size_t track_index = 0; track_index < track_count; ++track_index) {
      for (std::size_t detection_index = 0; detection_index < detection_count; ++detection_index) {
        const std::size_t flat = track_index * detection_count + detection_index;
        const std::optional<double> distance = filter.mahalanobis_distance_squared(
            tracks[track_index], detections_world[detection_index]);
        if (!distance.has_value() || !std::isfinite(*distance) || *distance > gate) {
          continue;
        }
        accepted_distances[flat] = *distance;
        costs.costs[flat] = *distance / gate;
      }
    }

    const std::vector<std::optional<std::size_t>> assigned = solve_cost_matrix(costs);
    AssociationResult result;
    std::vector<bool> detection_matched(detection_count, false);
    result.matches.reserve(std::min(track_count, detection_count));
    for (std::size_t track_index = 0; track_index < track_count; ++track_index) {
      const std::optional<std::size_t>& detection_index = assigned[track_index];
      if (!detection_index.has_value()) {
        result.unmatched_tracks.push_back(track_index);
        continue;
      }
      detection_matched[*detection_index] = true;
      TrackDetectionMatch match;
      match.track_index = track_index;
      match.detection_index = *detection_index;
      match.mahalanobis_distance_sq =
          accepted_distances[track_index * detection_count + *detection_index];
      result.matches.push_back(match);
    }
    for (std::size_t detection_index = 0; detection_index < detection_count; ++detection_index) {
      if (!detection_matched[detection_index]) {
        result.unmatched_detections.push_back(detection_index);
      }
    }
    return result;
  }

  /// Minimum-cardinality-safe assignment of a rectangular cost matrix.
  ///
  /// Returns one entry per track. nullopt means that track is unassigned. A
  /// returned detection index is unique, admissible, and part of a maximum-
  /// cardinality minimum-cost matching. costs must have exactly
  /// track_count * detection_count entries. Every present cost must be finite
  /// and greater than or equal to zero; any optional is forbidden.
  ///
  /// Present costs are used exactly as supplied. The caller is responsible for
  /// the normalization and dummy-penalty construction when cardinality priority
  /// is required. This entry point still adds dummy columns internally with
  /// penalty min(track_count, detection_count) + 1, which preserves maximum
  /// cardinality when every admissible cost is in [0, 1].
  static std::vector<std::optional<std::size_t>> solve_cost_matrix(const CostMatrix& costs) {
    if (costs.costs.size() != costs.track_count * costs.detection_count) {
      throw std::invalid_argument(
          "CostMatrix::costs size must equal track_count * detection_count");
    }
    for (const std::optional<double>& cost : costs.costs) {
      if (cost.has_value() && (!std::isfinite(*cost) || *cost < 0.0)) {
        throw std::invalid_argument("CostMatrix costs must be finite and nonnegative");
      }
    }

    std::vector<std::optional<std::size_t>> assignment(costs.track_count);
    if (costs.track_count == 0 || costs.detection_count == 0) {
      return assignment;
    }

    const std::size_t track_count = costs.track_count;
    const std::size_t detection_count = costs.detection_count;
    const std::size_t column_count = detection_count + track_count;
    const double penalty =
        static_cast<double>(std::min(track_count, detection_count) + 1U);

    // Row and column zero are sentinels. A zero match or predecessor means the
    // algorithm index has not been linked yet.
    std::vector<double> row_potential(track_count + 1U, 0.0);
    std::vector<double> column_potential(column_count + 1U, 0.0);
    std::vector<std::size_t> row_match(track_count + 1U, 0U);
    std::vector<std::size_t> column_match(column_count + 1U, 0U);
    std::vector<double> column_slack(column_count + 1U, 0.0);
    std::vector<std::size_t> slack_from_row(column_count + 1U, 0U);
    std::vector<char> scanned(column_count + 1U, 0);

    for (std::size_t free_row = 1; free_row <= track_count; ++free_row) {
      std::fill(column_slack.begin(), column_slack.end(),
                std::numeric_limits<double>::infinity());
      std::fill(slack_from_row.begin(), slack_from_row.end(), 0U);
      std::fill(scanned.begin(), scanned.end(), 0);

      std::vector<std::size_t> frontier;
      frontier.push_back(free_row);
      std::size_t frontier_begin = 0;
      std::size_t exposed_column = 0;

      // Dijkstra over alternating paths. Only columns whose reduced slack has
      // reached zero are scanned, so every augmenting path is shortest in the
      // feasible reduced-cost graph. Forbidden edges are never inserted.
      while (exposed_column == 0U) {
        if (frontier_begin == frontier.size()) {
          double delta = std::numeric_limits<double>::infinity();
          for (std::size_t column = 1; column <= column_count; ++column) {
            if (scanned[column] == 0 && column_slack[column] < delta) {
              delta = column_slack[column];
            }
          }
          if (!std::isfinite(delta)) {
            throw std::logic_error("DataAssociation failed to find a feasible assignment");
          }
          for (std::size_t index = 0; index < frontier.size(); ++index) {
            row_potential[frontier[index]] += delta;
          }
          for (std::size_t column = 1; column <= column_count; ++column) {
            if (scanned[column] != 0) {
              column_potential[column] -= delta;
            } else {
              column_slack[column] -= delta;
            }
          }
        }

        while (frontier_begin < frontier.size() && exposed_column == 0U) {
          const std::size_t row = frontier[frontier_begin];
          ++frontier_begin;
          for (std::size_t column = 1; column <= column_count; ++column) {
            if (scanned[column] != 0) {
              continue;
            }
            const std::optional<double> edge = edge_cost(
                costs, row - 1U, column - 1U, detection_count, penalty);
            if (!edge.has_value()) {
              continue;
            }
            const double reduced = *edge - row_potential[row] - column_potential[column];
            if (!(reduced < column_slack[column])) {
              continue;
            }
            column_slack[column] = reduced;
            slack_from_row[column] = row;
          }
        }

        for (std::size_t column = 1; column <= column_count && exposed_column == 0U; ++column) {
          if (scanned[column] != 0 || !(column_slack[column] <= 0.0) ||
              slack_from_row[column] == 0U) {
            continue;
          }
          scanned[column] = 1;
          const std::size_t matched_row = column_match[column];
          if (matched_row == 0U) {
            exposed_column = column;
          } else {
            frontier.push_back(matched_row);
          }
        }
      }

      for (std::size_t column = exposed_column; column != 0U;) {
        const std::size_t row = slack_from_row[column];
        const std::size_t previous_column = row_match[row];
        row_match[row] = column;
        column_match[column] = row;
        column = previous_column;
      }
    }

    for (std::size_t row = 1; row <= track_count; ++row) {
      const std::size_t column = row_match[row];
      if (column >= 1U && column <= detection_count) {
        assignment[row - 1U] = column - 1U;
      }
    }
    return assignment;
  }

 private:
  static std::optional<double> edge_cost(const CostMatrix& costs, std::size_t row,
                                         std::size_t column, std::size_t real_columns,
                                         double penalty) {
    if (column >= real_columns) {
      // Each track owns one dummy column. Matching another track to this dummy
      // would make two tracks share a non-assignment and is forbidden.
      if (column - real_columns == row) {
        return penalty;
      }
      return std::nullopt;
    }
    return costs.costs[row * costs.detection_count + column];
  }

  DataAssociationConfig config_;
};

}  // namespace prediction
