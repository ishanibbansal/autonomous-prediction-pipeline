#pragma once

#include "prediction/types.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

namespace prediction {

/// Noise and birth covariance for a single-target 6D linear Kalman filter.
///
/// Process noise is discrete piecewise-constant white acceleration noise, not
/// continuous-time white acceleration noise. During one predict interval the
/// unobserved linear acceleration (and yaw acceleration) is treated as one
/// constant draw. The resulting blocks are
///
///   Q_pos,pos = (1/4) dt^4 sigma^2
///   Q_pos,vel = (1/2) dt^3 sigma^2
///   Q_vel,vel = dt^2     sigma^2
///
/// with sigma^2 equal to sigma_a_sq on each translational axis and
/// sigma_alpha_sq on the yaw axis. Units are m^2/s^4 and rad^2/s^4.
/// Subdividing a horizon into smaller predict steps changes the accumulated
/// process noise: this model is not invariant to step subdivision.
///
/// This struct is a value. It is not validated until it is copied into a
/// KalmanFilter. A caller may build an invalid config and repair it before
/// construction; the filter then keeps its own validated copy.
struct KalmanFilterConfig {
  double sigma_a_sq{1.0};              // m^2/s^4, translational acceleration variance
  double sigma_alpha_sq{0.5};          // rad^2/s^4, yaw-acceleration variance
  double sigma_pos_sq{0.25};           // m^2, world position measurement variance
  double sigma_psi_sq{0.05};           // rad^2, world heading measurement variance
  double initial_position_variance{1.0};     // m^2
  double initial_velocity_variance{100.0};   // m^2/s^2
  double initial_yaw_variance{0.1};          // rad^2
  double initial_turn_rate_variance{1.0};    // rad^2/s^2
};

/// Single-target linear Kalman filter on TrackState.
///
/// State is [x, y, v_x, v_y, psi, omega]^T in the global world frame. psi is a
/// directed 2*pi-periodic heading in radians, not a pi-ambiguous box orientation.
/// The two translational axes and the (psi, omega) axis are independent linear
/// blocks. This is not a coordinated-turn (CTRV) model: a nonzero turn rate does
/// not rotate (v_x, v_y). Velocity direction and heading are independent.
///
/// Observations are world-frame Detection values [x, y, heading]. Detection does
/// not encode its frame, so the caller must already have transformed the
/// measurement into the world frame. predict and update never change track_id
/// or status. update does not gate: a statistically distant but otherwise valid
/// measurement is still applied. Association thresholds belong to the caller.
///
/// Failure of predict, update, and mahalanobis_distance_squared is transactional.
/// Any failed input, admissibility, factorization, or post-computation finiteness
/// check leaves the track untouched and returns false or std::nullopt.
class KalmanFilter {
 public:
  /// Absolute-plus-relative bound used when judging covariance symmetry and
  /// when allowing a tiny negative eigenvalue of an otherwise PSD matrix:
  /// |a - b| <= kAdmissibilityTolerance * (1 + max(|a|, |b|)), and
  /// lambda_min >= -kAdmissibilityTolerance * (1 + lambda_max).
  static constexpr double kAdmissibilityTolerance = 1e-9;

  /// Negative Mahalanobis distances no smaller than this are treated as
  /// rounding noise and clamped to 0. More negative results are rejected.
  static constexpr double kMahalanobisClampTolerance = 1e-12;

  explicit KalmanFilter(KalmanFilterConfig config = {}) : config_(std::move(config)) {
    validate_variance(config_.sigma_a_sq, "sigma_a_sq");
    validate_variance(config_.sigma_alpha_sq, "sigma_alpha_sq");
    validate_variance(config_.sigma_pos_sq, "sigma_pos_sq");
    validate_variance(config_.sigma_psi_sq, "sigma_psi_sq");
    validate_variance(config_.initial_position_variance, "initial_position_variance");
    validate_variance(config_.initial_velocity_variance, "initial_velocity_variance");
    validate_variance(config_.initial_yaw_variance, "initial_yaw_variance");
    validate_variance(config_.initial_turn_rate_variance, "initial_turn_rate_variance");
  }

  const KalmanFilterConfig& config() const { return config_; }

  /// Birth from one world-frame detection. Velocity and turn rate start at 0.
  /// Returns nullopt when det_world fails the shared detection checks.
  std::optional<TrackState> initialize(
      const Detection& det_world, std::uint64_t track_id,
      TrackStatus status = TrackStatus::Tentative) const {
    if (!detection_is_valid(det_world)) {
      return std::nullopt;
    }
    TrackState track;
    track.track_id = track_id;
    track.status = status;
    track.size = det_world.size;
    track.state << det_world.position.x(), det_world.position.y(), 0.0, 0.0,
        normalize_angle(det_world.heading), 0.0;
    track.covariance.setZero();
    track.covariance(0, 0) = config_.initial_position_variance;
    track.covariance(1, 1) = config_.initial_position_variance;
    track.covariance(2, 2) = config_.initial_velocity_variance;
    track.covariance(3, 3) = config_.initial_velocity_variance;
    track.covariance(4, 4) = config_.initial_yaw_variance;
    track.covariance(5, 5) = config_.initial_turn_rate_variance;
    return track;
  }

  /// Constant-velocity / constant-turn-rate time update with piecewise-constant
  /// white acceleration noise. dt == 0 on an admissible track is an exact no-op.
  bool predict(TrackState& track, double dt) const {
    if (!dt_is_valid(dt) || !track_is_admissible(track)) {
      return false;
    }
    if (dt == 0.0) {
      return true;
    }

    const StateVector prior_state = track.state;
    const Covariance prior_covariance = track.covariance;
    const Eigen::Vector2d prior_size = track.size;

    StateVector predicted = prior_state;
    predicted(0) += prior_state(2) * dt;
    predicted(1) += prior_state(3) * dt;
    predicted(4) = normalize_angle(prior_state(4) + prior_state(5) * dt);

    const Transition F = transition_matrix(dt);
    Covariance predicted_covariance =
        F * prior_covariance * F.transpose() + process_noise(dt);
    symmetrize(predicted_covariance);

    if (!state_is_finite(predicted) || !matrix_is_finite(predicted_covariance)) {
      track.state = prior_state;
      track.covariance = prior_covariance;
      track.size = prior_size;
      return false;
    }
    track.state = predicted;
    track.covariance = predicted_covariance;
    return true;
  }

  /// Squared Mahalanobis distance of a world-frame detection. Heading residual
  /// is wrapped into [-pi, pi]. Does not modify the track.
  std::optional<double> mahalanobis_distance_squared(const TrackState& track,
                                                     const Detection& det_world) const {
    Innovation innovation;
    if (!prepare_innovation(track, det_world, innovation)) {
      return std::nullopt;
    }
    const Eigen::Vector3d solved = innovation.ldlt.solve(innovation.residual);
    if (!solved.allFinite()) {
      return std::nullopt;
    }
    const double distance = innovation.residual.dot(solved);
    if (!std::isfinite(distance)) {
      return std::nullopt;
    }
    if (distance < 0.0) {
      if (distance >= -kMahalanobisClampTolerance) {
        return 0.0;
      }
      return std::nullopt;
    }
    return distance;
  }

  /// Measurement update. Does not reject large innovations; the caller gates.
  /// On success, size is replaced by the detection size. id and status stay.
  bool update(TrackState& track, const Detection& det_world) const {
    Innovation innovation;
    if (!prepare_innovation(track, det_world, innovation)) {
      return false;
    }

    const StateVector prior_state = track.state;
    const Covariance prior_covariance = track.covariance;
    const Eigen::Vector2d prior_size = track.size;

    const Eigen::Matrix<double, 6, 3> pht = prior_covariance * measurement_matrix().transpose();
    const Eigen::Matrix<double, 3, 6> solved = innovation.ldlt.solve(pht.transpose());
    if (!solved.allFinite()) {
      return false;
    }
    const Eigen::Matrix<double, 6, 3> gain = solved.transpose();
    if (!gain.allFinite()) {
      return false;
    }

    StateVector posterior = prior_state + gain * innovation.residual;
    posterior(4) = normalize_angle(posterior(4));

    const Covariance identity_minus_kh =
        Covariance::Identity() - gain * measurement_matrix();
    Covariance posterior_covariance =
        identity_minus_kh * prior_covariance * identity_minus_kh.transpose() +
        gain * measurement_noise() * gain.transpose();
    symmetrize(posterior_covariance);

    if (!state_is_finite(posterior) || !matrix_is_finite(posterior_covariance)) {
      track.state = prior_state;
      track.covariance = prior_covariance;
      track.size = prior_size;
      return false;
    }
    track.state = posterior;
    track.covariance = posterior_covariance;
    track.size = det_world.size;
    return true;
  }

 private:
  using StateVector = Eigen::Matrix<double, 6, 1>;
  using Covariance = Eigen::Matrix<double, 6, 6>;
  using Transition = Eigen::Matrix<double, 6, 6>;

  struct Innovation {
    Eigen::Vector3d residual{Eigen::Vector3d::Zero()};
    Eigen::LDLT<Eigen::Matrix3d> ldlt{};
  };

  static void validate_variance(double value, const char* name) {
    if (!std::isfinite(value) || !(value > 0.0)) {
      throw std::invalid_argument(std::string("KalmanFilterConfig::") + name +
                                  " must be finite and strictly positive");
    }
  }

  static bool finite_positive(double value) { return std::isfinite(value) && value > 0.0; }

  static bool detection_is_valid(const Detection& detection) {
    return detection.position.allFinite() && std::isfinite(detection.heading) &&
           std::isfinite(detection.score) && finite_positive(detection.size.x()) &&
           finite_positive(detection.size.y());
  }

  static bool dt_is_valid(double dt) { return std::isfinite(dt) && dt >= 0.0; }

  static bool state_is_finite(const StateVector& state) { return state.allFinite(); }

  static bool matrix_is_finite(const Covariance& matrix) { return matrix.allFinite(); }

  static bool nearly_equal(double lhs, double rhs) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= kAdmissibilityTolerance * (1.0 + scale);
  }

  static bool covariance_is_symmetric(const Covariance& covariance) {
    for (int row = 0; row < covariance.rows(); ++row) {
      for (int col = row + 1; col < covariance.cols(); ++col) {
        if (!nearly_equal(covariance(row, col), covariance(col, row))) {
          return false;
        }
      }
    }
    return true;
  }

  /// Symmetric input is accepted as PSD when every eigenvalue is nonnegative
  /// up to a scale-aware residual. A strictly singular covariance is allowed.
  static bool covariance_is_positive_semidefinite(const Covariance& covariance) {
    const Eigen::SelfAdjointEigenSolver<Covariance> solver(covariance);
    if (solver.info() != Eigen::Success) {
      return false;
    }
    const auto& eigenvalues = solver.eigenvalues();
    if (!eigenvalues.allFinite()) {
      return false;
    }
    const double largest = std::max(0.0, eigenvalues.maxCoeff());
    const double floor = -kAdmissibilityTolerance * (1.0 + largest);
    return eigenvalues.minCoeff() >= floor;
  }

  static bool track_is_admissible(const TrackState& track) {
    return state_is_finite(track.state) && matrix_is_finite(track.covariance) &&
           track.size.allFinite() && finite_positive(track.size.x()) &&
           finite_positive(track.size.y()) && covariance_is_symmetric(track.covariance) &&
           covariance_is_positive_semidefinite(track.covariance);
  }

  static void symmetrize(Covariance& covariance) {
    covariance = 0.5 * (covariance + covariance.transpose());
  }

  static Eigen::Matrix<double, 3, 6> measurement_matrix() {
    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 4) = 1.0;
    return H;
  }

  Eigen::Matrix3d measurement_noise() const {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, 0) = config_.sigma_pos_sq;
    R(1, 1) = config_.sigma_pos_sq;
    R(2, 2) = config_.sigma_psi_sq;
    return R;
  }

  static Transition transition_matrix(double dt) {
    Transition F = Transition::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;
    F(4, 5) = dt;
    return F;
  }

  Covariance process_noise(double dt) const {
    Covariance Q = Covariance::Zero();
    fill_noise_block(Q, 0, 2, dt, config_.sigma_a_sq);
    fill_noise_block(Q, 1, 3, dt, config_.sigma_a_sq);
    fill_noise_block(Q, 4, 5, dt, config_.sigma_alpha_sq);
    return Q;
  }

  static void fill_noise_block(Covariance& Q, int position_index, int velocity_index,
                               double dt, double variance) {
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;
    Q(position_index, position_index) = 0.25 * dt4 * variance;
    Q(position_index, velocity_index) = 0.5 * dt3 * variance;
    Q(velocity_index, position_index) = 0.5 * dt3 * variance;
    Q(velocity_index, velocity_index) = dt2 * variance;
  }

  /// Positive-definite innovation covariance. LDLT success alone is not enough:
  /// zero pivots (semidefinite S) are rejected.
  static bool innovation_covariance_is_positive_definite(const Eigen::LDLT<Eigen::Matrix3d>& ldlt) {
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
      return false;
    }
    const auto pivots = ldlt.vectorD();
    return pivots.allFinite() && pivots.minCoeff() > 0.0;
  }

  bool prepare_innovation(const TrackState& track, const Detection& det_world,
                          Innovation& innovation) const {
    if (!track_is_admissible(track) || !detection_is_valid(det_world)) {
      return false;
    }
    const Eigen::Vector3d measurement(det_world.position.x(), det_world.position.y(),
                                      det_world.heading);
    const Eigen::Matrix<double, 3, 6> H = measurement_matrix();
    innovation.residual = measurement - H * track.state;
    innovation.residual(2) = normalize_angle(det_world.heading - track.state(4));
    if (!innovation.residual.allFinite()) {
      return false;
    }
    const Eigen::Matrix3d S = H * track.covariance * H.transpose() + measurement_noise();
    if (!S.allFinite()) {
      return false;
    }
    innovation.ldlt.compute(S);
    return innovation_covariance_is_positive_definite(innovation.ldlt);
  }

  KalmanFilterConfig config_;
};

}  // namespace prediction
