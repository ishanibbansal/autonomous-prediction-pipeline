#pragma once

#include "prediction/kalman_filter.hpp"
#include "prediction/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace prediction {

/// Forecast horizon, mixture weights, and heuristic position diffusion.
///
/// This struct is a value. It is not validated until it is copied into a
/// KinematicTrajectoryPredictor. A caller may build an invalid config and repair
/// it before construction; the predictor then keeps its own validated copy.
/// Probabilities that already sum to 1 within kProbabilitySumTolerance are
/// stored after dividing by that sum, so config() can differ from the value
/// passed to the constructor by a relative residual no larger than that
/// tolerance.
struct PredictorConfig {
  double horizon_seconds{3.0};     // Total requested horizon H, seconds. > 0.
  double step_size_seconds{0.1};   // Future-step spacing dt, seconds. > 0.
  double branch_turn_rate{0.15};   // Minimum |signed turn rate| of a turn mode, rad/s. > 0.
  double prob_straight{0.60};      // Straight-mode mixture weight in (0, 1].
  double prob_left{0.20};          // Counterclockwise-mode mixture weight in (0, 1].
  double prob_right{0.20};         // Clockwise-mode mixture weight in (0, 1].
  double pos_diffusion_rate{0.25}; // Position-variance diffusion, m^2/s. >= 0. Not squared again.
};

/// Constant-speed, three-mode kinematic forecast of one world-frame track.
///
/// State semantics match KalmanFilter: world velocity (v_x, v_y) is independent
/// of body heading psi. A mode with signed rate r therefore rotates the supplied
/// world velocity; it does not replace that velocity by speed along heading.
/// Positive r is counterclockwise in the world frame, not "overtake", and
/// negative r is clockwise, not "yield". Straight translation is p0 + t * v0
/// and does not use heading at all.
///
/// These are geometric hypotheses, not a constant-turn-rate-and-acceleration
/// model and not a calibrated behavior predictor. Heading and velocity are
/// rotated by the same hypothesis rate, which preserves their initial angular
/// offset. An exactly stationary track is the exception: every mode holds its
/// position and normalized heading and emits zero velocity, even if omega is
/// nonzero. Any nonzero finite velocity is rolled out; there is no speed floor.
///
/// Samples are future steps only. Sample m = 1..N has offset
/// t_m = m * dt, with N = round(H / dt). The realized horizon is N * dt, which
/// may overshoot or undershoot H by up to about dt/2. There is no partial final
/// step and the current state is not repeated as a waypoint. The caller must
/// already have made the track valid at current_timestamp. TrackState has no
/// timestamp, and this predictor does not coast a Lost track a second time.
///
/// Position covariance is the same in every mode:
///   C(t) = P_pp + t^2 P_vv + q t I.
/// q is pos_diffusion_rate in m^2/s. This is a mode-independent uncertainty
/// envelope. It ignores position-velocity cross covariance and uncertainty in
/// heading and turn rate. It is not an EKF covariance, not a CTRV covariance,
/// and not a statistical upper bound. Diagonals are nondecreasing in exact
/// arithmetic when the input blocks are PSD, but they need not grow strictly,
/// and a singular input can remain singular.
///
/// The constructor throws std::invalid_argument for an invalid configuration.
/// predict throws std::invalid_argument for a nonfinite timestamp, a nonfinite
/// or nonpositive size, or a nonfinite, materially asymmetric, or materially
/// indefinite covariance. A computed position, velocity, heading, covariance, or
/// timestamp that is nonfinite, or a timestamp grid that is finite but no longer
/// strictly increasing, throws std::overflow_error. Allocation failure propagates
/// as std::bad_alloc. A failed single or batch call returns nothing, and the
/// arguments are not modified.
///
/// predict returns by value and may allocate on every call. Output size is
/// O(N) per track and O(M N) for a batch of M tracks, with the mode count fixed
/// at three. There is no scratch-workspace overload in this phase.
class KinematicTrajectoryPredictor {
 public:
  /// Hard cap on N so one track cannot request an unbounded waypoint allocation.
  /// This does not cap the number of tracks in a batch.
  static constexpr std::size_t kMaxPredictionSteps = 10000U;

  /// |r t / 2| below this uses the sinc Taylor series. At and above it, sinc
  /// uses sin(z) / z.
  static constexpr double kSincSeriesThreshold = 1e-4;

  /// Accepted residual of the three mixture weights around 1.
  static constexpr double kProbabilitySumTolerance = 1e-6;

  /// Same absolute-plus-relative covariance tolerance as KalmanFilter.
  static constexpr double kAdmissibilityTolerance = KalmanFilter::kAdmissibilityTolerance;

  explicit KinematicTrajectoryPredictor(PredictorConfig config = {})
      : KinematicTrajectoryPredictor(prepare(std::move(config))) {}

  /// Forecast one world-frame track. current_timestamp is the time at which
  /// track.state is already valid. Throws on invalid input or a nonrepresentable
  /// forecast; the track is not modified.
  PredictedTrajectory predict(const TrackState& track, double current_timestamp) const {
    validate_timestamp(current_timestamp, std::nullopt);
    validate_track(track, std::nullopt);
    return rollout(track, current_timestamp, std::nullopt);
  }

  /// Forecast every track, preserving input order and cardinality.
  /// An empty range is valid and returns an empty vector. One invalid track or
  /// a nonrepresentable sample rejects the whole call; no partial vector is
  /// returned. Duplicate and unsorted track IDs are preserved.
  std::vector<PredictedTrajectory> predict(const std::vector<TrackState>& tracks,
                                           double current_timestamp) const {
    validate_timestamp(current_timestamp, std::nullopt);
    for (std::size_t index = 0; index < tracks.size(); ++index) {
      validate_track(tracks[index], index);
    }
    std::vector<PredictedTrajectory> forecasts;
    forecasts.reserve(tracks.size());
    for (std::size_t index = 0; index < tracks.size(); ++index) {
      forecasts.push_back(rollout(tracks[index], current_timestamp, index));
    }
    return forecasts;
  }

  const PredictorConfig& config() const { return config_; }

 private:
  struct Prepared {
    PredictorConfig config{};
    std::size_t step_count{0U};
  };

  explicit KinematicTrajectoryPredictor(Prepared prepared)
      : config_(std::move(prepared.config)), step_count_(prepared.step_count) {}

  using StateCovariance = Eigen::Matrix<double, 6, 6>;

  static Prepared prepare(PredictorConfig config) {
    require_finite_positive(config.horizon_seconds, "PredictorConfig::horizon_seconds");
    require_finite_positive(config.step_size_seconds, "PredictorConfig::step_size_seconds");
    if (!(config.horizon_seconds >= config.step_size_seconds)) {
      throw std::invalid_argument(
          "PredictorConfig::horizon_seconds must be >= PredictorConfig::step_size_seconds");
    }
    require_finite_positive(config.branch_turn_rate, "PredictorConfig::branch_turn_rate");
    require_probability(config.prob_straight, "PredictorConfig::prob_straight");
    require_probability(config.prob_left, "PredictorConfig::prob_left");
    require_probability(config.prob_right, "PredictorConfig::prob_right");

    const double probability_sum = config.prob_straight + config.prob_left + config.prob_right;
    if (!std::isfinite(probability_sum) ||
        std::abs(probability_sum - 1.0) > kProbabilitySumTolerance) {
      throw std::invalid_argument(
          "PredictorConfig probabilities must sum to 1 within 1e-6");
    }
    config.prob_straight /= probability_sum;
    config.prob_left /= probability_sum;
    config.prob_right /= probability_sum;

    if (!std::isfinite(config.pos_diffusion_rate) || config.pos_diffusion_rate < 0.0) {
      throw std::invalid_argument(
          "PredictorConfig::pos_diffusion_rate must be finite and >= 0");
    }

    const double quotient = config.horizon_seconds / config.step_size_seconds;
    if (!std::isfinite(quotient)) {
      throw std::invalid_argument(
          "PredictorConfig::horizon_seconds / PredictorConfig::step_size_seconds must be finite");
    }
    const double rounded = std::round(quotient);
    if (!std::isfinite(rounded) || rounded < 1.0 ||
        rounded > static_cast<double>(kMaxPredictionSteps)) {
      throw std::invalid_argument(
          "PredictorConfig prediction step count must be in [1, KinematicTrajectoryPredictor::kMaxPredictionSteps]");
    }
    Prepared prepared;
    prepared.config = config;
    prepared.step_count = static_cast<std::size_t>(rounded);
    return prepared;
  }

  static void require_finite_positive(double value, const char* name) {
    if (!std::isfinite(value) || !(value > 0.0)) {
      throw std::invalid_argument(std::string(name) + " must be finite and strictly positive");
    }
  }

  static void require_probability(double value, const char* name) {
    if (!std::isfinite(value) || !(value > 0.0) || value > 1.0) {
      throw std::invalid_argument(std::string(name) + " must be finite and in (0, 1]");
    }
  }

  static std::string track_context(const TrackState& track, std::optional<std::size_t> batch_index) {
    std::string context = "track_id " + std::to_string(track.track_id);
    if (batch_index.has_value()) {
      context += ", batch index " + std::to_string(*batch_index);
    }
    return context;
  }

  static void validate_timestamp(double current_timestamp, std::optional<std::size_t> batch_index) {
    if (std::isfinite(current_timestamp)) {
      return;
    }
    std::string message = "current_timestamp must be finite";
    if (batch_index.has_value()) {
      message += " (batch index " + std::to_string(*batch_index) + ")";
    }
    throw std::invalid_argument(message);
  }

  static bool finite_positive(double value) { return std::isfinite(value) && value > 0.0; }

  static bool nearly_equal(double lhs, double rhs) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= kAdmissibilityTolerance * (1.0 + scale);
  }

  static bool covariance_is_symmetric(const StateCovariance& covariance) {
    for (int row = 0; row < covariance.rows(); ++row) {
      for (int col = row + 1; col < covariance.cols(); ++col) {
        if (!nearly_equal(covariance(row, col), covariance(col, row))) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename Derived>
  static bool eigenvalues_are_psd(const Eigen::MatrixBase<Derived>& eigenvalues) {
    if (!eigenvalues.allFinite()) {
      return false;
    }
    const double largest = std::max(0.0, eigenvalues.maxCoeff());
    const double floor = -kAdmissibilityTolerance * (1.0 + largest);
    return eigenvalues.minCoeff() >= floor;
  }

  static void validate_track(const TrackState& track, std::optional<std::size_t> batch_index) {
    const std::string where = track_context(track, batch_index);
    if (!track.state.allFinite()) {
      throw std::invalid_argument("TrackState::state must be finite (" + where + ")");
    }
    if (!track.covariance.allFinite()) {
      throw std::invalid_argument("TrackState::covariance must be finite (" + where + ")");
    }
    if (!track.size.allFinite() || !finite_positive(track.size.x()) ||
        !finite_positive(track.size.y())) {
      throw std::invalid_argument(
          "TrackState::size components must be finite and strictly positive (" + where + ")");
    }
    if (!covariance_is_symmetric(track.covariance)) {
      throw std::invalid_argument("TrackState::covariance must be symmetric (" + where + ")");
    }
    const Eigen::SelfAdjointEigenSolver<StateCovariance> solver(track.covariance);
    if (solver.info() != Eigen::Success || !eigenvalues_are_psd(solver.eigenvalues())) {
      throw std::invalid_argument(
          "TrackState::covariance must be positive semidefinite (" + where + ")");
    }
  }

  /// Copy the covariance and lift only tolerance-level negative eigenvalues to
  /// zero. A covariance that failed validate_track is not repaired here.
  static StateCovariance sanitized_covariance(const TrackState& track,
                                              const std::string& where) {
    const Eigen::SelfAdjointEigenSolver<StateCovariance> solver(track.covariance);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
      throw std::overflow_error("TrackState::covariance eigensolve failed (" + where + ")");
    }
    const auto& eigenvalues = solver.eigenvalues();
    if (eigenvalues.minCoeff() >= 0.0) {
      return track.covariance;
    }
    Eigen::Matrix<double, 6, 1> clamped = eigenvalues;
    for (int index = 0; index < clamped.size(); ++index) {
      if (clamped(index) < 0.0) {
        clamped(index) = 0.0;
      }
    }
    return solver.eigenvectors() * clamped.asDiagonal() * solver.eigenvectors().transpose();
  }

  static Eigen::Matrix2d symmetrize(const Eigen::Matrix2d& matrix) {
    return (0.5 * (matrix + matrix.transpose())).eval();
  }

  /// C(t) as specified, symmetrized and then clamped if rounding leaves a tiny
  /// negative eigenvalue. A material failure is overflow, not silent repair.
  static Eigen::Matrix2d forecast_covariance(const Eigen::Matrix2d& position_block,
                                             const Eigen::Matrix2d& velocity_block, double offset,
                                             double diffusion, const std::string& where) {
    const Eigen::Matrix2d forecast =
        symmetrize(position_block + offset * offset * velocity_block +
                   diffusion * offset * Eigen::Matrix2d::Identity());
    if (!forecast.allFinite()) {
      throw std::overflow_error("forecast position covariance is non-finite (" + where + ")");
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(forecast);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
        !eigenvalues_are_psd(solver.eigenvalues())) {
      throw std::overflow_error("forecast position covariance is not positive semidefinite (" +
                                where + ")");
    }
    if (solver.eigenvalues().minCoeff() >= 0.0) {
      return forecast;
    }
    Eigen::Vector2d clamped = solver.eigenvalues();
    for (int index = 0; index < clamped.size(); ++index) {
      if (clamped(index) < 0.0) {
        clamped(index) = 0.0;
      }
    }
    return symmetrize(solver.eigenvectors() * clamped.asDiagonal() *
                      solver.eigenvectors().transpose());
  }

  static double sinc(double angle) {
    if (std::abs(angle) < kSincSeriesThreshold) {
      const double angle_sq = angle * angle;
      return 1.0 - angle_sq / 6.0 + angle_sq * angle_sq / 120.0;
    }
    return std::sin(angle) / angle;
  }

  /// Integral of R(r s) from 0 to t, applied to v0. Valid for r = 0 as t * I.
  static Eigen::Vector2d integrated_rotation(const Eigen::Vector2d& velocity, double rate,
                                             double offset) {
    const double half_turn = 0.5 * rate * offset;
    const double scale = offset * sinc(half_turn);
    const double cosine = std::cos(half_turn);
    const double sine = std::sin(half_turn);
    Eigen::Vector2d displacement;
    displacement.x() = scale * (cosine * velocity.x() - sine * velocity.y());
    displacement.y() = scale * (sine * velocity.x() + cosine * velocity.y());
    return displacement;
  }

  static Eigen::Vector2d rotate(const Eigen::Vector2d& velocity, double turn) {
    const double cosine = std::cos(turn);
    const double sine = std::sin(turn);
    Eigen::Vector2d rotated;
    rotated.x() = cosine * velocity.x() - sine * velocity.y();
    rotated.y() = sine * velocity.x() + cosine * velocity.y();
    return rotated;
  }

  static void require_finite_vector(const Eigen::Vector2d& vector, const char* what,
                                    const TrackState& track, std::optional<std::size_t> batch_index) {
    if (!vector.allFinite()) {
      throw std::overflow_error(std::string(what) + " is non-finite (" +
                                track_context(track, batch_index) + ")");
    }
  }

  static PredictedWaypoint legacy_from(const TrajectoryPoint& point) {
    PredictedWaypoint legacy;
    legacy.timestamp = point.timestamp;
    legacy.position = point.position;
    legacy.heading = point.heading;
    legacy.position_covariance = point.position_covariance;
    return legacy;
  }

  PredictedTrajectory rollout(const TrackState& track, double current_timestamp,
                              std::optional<std::size_t> batch_index) const {
    const std::string where = track_context(track, batch_index);
    const Eigen::Vector2d position = track.state.head<2>();
    const Eigen::Vector2d velocity = track.state.segment<2>(2);
    const double heading0 = track.state(4);
    const double omega = track.state(5);
    const double normalized_heading = normalize_angle(heading0);
    if (!std::isfinite(normalized_heading)) {
      throw std::overflow_error("normalized heading is non-finite (" + where + ")");
    }

    const bool stationary = velocity.x() == 0.0 && velocity.y() == 0.0;
    const double left_rate = std::max(omega, config_.branch_turn_rate);
    const double right_rate = std::min(omega, -config_.branch_turn_rate);
    const double rates[3] = {0.0, left_rate, right_rate};

    const StateCovariance covariance = sanitized_covariance(track, where);
    const Eigen::Matrix2d position_block = symmetrize(covariance.block<2, 2>(0, 0));
    const Eigen::Matrix2d velocity_block = symmetrize(covariance.block<2, 2>(2, 2));

    PredictedTrajectory forecast;
    forecast.track_id = track.track_id;
    forecast.current_state = track;
    forecast.waypoints.reserve(step_count_);
    forecast.modes.resize(3U);
    const double probabilities[3] = {config_.prob_straight, config_.prob_left, config_.prob_right};
    for (std::size_t mode = 0; mode < 3U; ++mode) {
      forecast.modes[mode].probability = probabilities[mode];
      forecast.modes[mode].waypoints.resize(step_count_);
    }

    for (std::size_t step = 1U; step <= step_count_; ++step) {
      const double offset = static_cast<double>(step) * config_.step_size_seconds;
      if (!std::isfinite(offset) || !(offset > 0.0)) {
        throw std::overflow_error("forecast time offset is non-finite (" + where + ")");
      }
      const double timestamp = current_timestamp + offset;
      if (!std::isfinite(timestamp)) {
        throw std::overflow_error("forecast timestamp is non-finite (" + where + ")");
      }
      const double previous =
          step == 1U ? current_timestamp
                     : current_timestamp +
                           static_cast<double>(step - 1U) * config_.step_size_seconds;
      if (!(timestamp > previous)) {
        throw std::overflow_error("forecast timestamps are not strictly increasing (" + where +
                                  ")");
      }

      const Eigen::Matrix2d position_covariance = forecast_covariance(
          position_block, velocity_block, offset, config_.pos_diffusion_rate, where);

      for (std::size_t mode = 0; mode < 3U; ++mode) {
        TrajectoryPoint point;
        point.timestamp = timestamp;
        point.position_covariance = position_covariance;
        if (stationary) {
          point.position = position;
          point.velocity.setZero();
          point.heading = normalized_heading;
        } else if (mode == 0U) {
          point.position = position + offset * velocity;
          point.velocity = velocity;
          point.heading = normalized_heading;
        } else {
          const double turn = rates[mode] * offset;
          if (!std::isfinite(turn)) {
            throw std::overflow_error("forecast turn angle is non-finite (" + where + ")");
          }
          point.position = position + integrated_rotation(velocity, rates[mode], offset);
          point.velocity = rotate(velocity, turn);
          point.heading = normalize_angle(normalized_heading + turn);
        }
        require_finite_vector(point.position, "forecast position", track, batch_index);
        require_finite_vector(point.velocity, "forecast velocity", track, batch_index);
        if (!std::isfinite(point.heading)) {
          throw std::overflow_error("forecast heading is non-finite (" + where + ")");
        }
        forecast.modes[mode].waypoints[step - 1U] = point;
      }
      forecast.waypoints.push_back(legacy_from(forecast.modes[0].waypoints[step - 1U]));
    }
    return forecast;
  }

  PredictorConfig config_;
  std::size_t step_count_{0U};
};

}  // namespace prediction
