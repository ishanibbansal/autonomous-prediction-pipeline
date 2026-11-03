#include "prediction/kalman_filter.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

namespace prediction {
namespace test_support {
const KalmanFilter& linkage_anchor();
}  // namespace test_support
}  // namespace prediction

namespace {

constexpr double kTolerance = 1e-9;
constexpr double kLooseTolerance = 1e-6;

prediction::Detection make_detection(double x, double y, double heading, double length = 4.5,
                                     double width = 1.8, double score = 0.9) {
  prediction::Detection detection;
  detection.position = Eigen::Vector2d(x, y);
  detection.size = Eigen::Vector2d(length, width);
  detection.heading = heading;
  detection.score = score;
  return detection;
}

prediction::KalmanFilterConfig test_config() {
  prediction::KalmanFilterConfig config;
  config.sigma_a_sq = 1.0;
  config.sigma_alpha_sq = 0.25;
  config.sigma_pos_sq = 0.25;
  config.sigma_psi_sq = 0.04;
  config.initial_position_variance = 1.0;
  config.initial_velocity_variance = 100.0;
  config.initial_yaw_variance = 0.1;
  config.initial_turn_rate_variance = 1.0;
  return config;
}

Eigen::Matrix<double, 6, 6> expected_process_noise(double dt, double sigma_a_sq,
                                                   double sigma_alpha_sq) {
  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt3 * dt;
  const auto fill = [&](int position, int velocity, double variance) {
    Q(position, position) = 0.25 * dt4 * variance;
    Q(position, velocity) = 0.5 * dt3 * variance;
    Q(velocity, position) = 0.5 * dt3 * variance;
    Q(velocity, velocity) = dt2 * variance;
  };
  fill(0, 2, sigma_a_sq);
  fill(1, 3, sigma_a_sq);
  fill(4, 5, sigma_alpha_sq);
  return Q;
}

Eigen::Matrix<double, 6, 6> transition(double dt) {
  Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
  F(0, 2) = dt;
  F(1, 3) = dt;
  F(4, 5) = dt;
  return F;
}

void expect_matrix_near(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected,
                        double tolerance) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (int row = 0; row < actual.rows(); ++row) {
    for (int col = 0; col < actual.cols(); ++col) {
      EXPECT_NEAR(actual(row, col), expected(row, col), tolerance)
          << "at (" << row << ", " << col << ")";
    }
  }
}

void expect_unchanged(const prediction::TrackState& actual,
                      const prediction::TrackState& prior) {
  EXPECT_EQ(actual.track_id, prior.track_id);
  EXPECT_EQ(actual.status, prior.status);
  expect_matrix_near(actual.size, prior.size, 0.0);
  expect_matrix_near(actual.state, prior.state, 0.0);
  expect_matrix_near(actual.covariance, prior.covariance, 0.0);
}

bool is_symmetric(const Eigen::MatrixXd& matrix) {
  return matrix.isApprox(matrix.transpose(), kTolerance);
}

}  // namespace

TEST(KalmanFilter, ConfigValidationRejectsNonPositiveAndNonFinite) {
  prediction::KalmanFilterConfig config = test_config();
  EXPECT_NO_THROW(prediction::KalmanFilter{config});

  const double invalid[] = {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()};
  double* fields[] = {&config.sigma_a_sq,
                      &config.sigma_alpha_sq,
                      &config.sigma_pos_sq,
                      &config.sigma_psi_sq,
                      &config.initial_position_variance,
                      &config.initial_velocity_variance,
                      &config.initial_yaw_variance,
                      &config.initial_turn_rate_variance};
  for (double* field : fields) {
    const double saved = *field;
    for (const double bad : invalid) {
      *field = bad;
      EXPECT_THROW(prediction::KalmanFilter{config}, std::invalid_argument);
    }
    *field = saved;
  }
}

TEST(KalmanFilter, InitializeFromWorldDetection) {
  const prediction::KalmanFilter filter(test_config());
  const double wrapped = prediction::normalize_angle(3.0 * prediction::kPi + 0.2);
  const auto track = filter.initialize(make_detection(10.0, -4.0, 3.0 * prediction::kPi + 0.2), 7,
                                        prediction::TrackStatus::Confirmed);
  ASSERT_TRUE(track.has_value());
  EXPECT_EQ(track->track_id, 7u);
  EXPECT_EQ(track->status, prediction::TrackStatus::Confirmed);
  EXPECT_NEAR(track->size.x(), 4.5, kTolerance);
  EXPECT_NEAR(track->size.y(), 1.8, kTolerance);
  EXPECT_NEAR(track->state(0), 10.0, kTolerance);
  EXPECT_NEAR(track->state(1), -4.0, kTolerance);
  EXPECT_NEAR(track->state(2), 0.0, kTolerance);
  EXPECT_NEAR(track->state(3), 0.0, kTolerance);
  EXPECT_NEAR(track->state(4), wrapped, kTolerance);
  EXPECT_NEAR(track->state(5), 0.0, kTolerance);

  Eigen::Matrix<double, 6, 6> expected = Eigen::Matrix<double, 6, 6>::Zero();
  expected.diagonal() << 1.0, 1.0, 100.0, 100.0, 0.1, 1.0;
  expect_matrix_near(track->covariance, expected, 0.0);

  prediction::Detection bad = make_detection(1.0, 2.0, 0.1);
  bad.size.y() = 0.0;
  EXPECT_FALSE(filter.initialize(bad, 1).has_value());
  bad = make_detection(1.0, 2.0, 0.1);
  bad.score = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(filter.initialize(bad, 1).has_value());
  // Score is not range-checked. A finite value outside [0, 1] is accepted.
  EXPECT_TRUE(filter.initialize(make_detection(1.0, 2.0, 0.1, 4.5, 1.8, 1.5), 3).has_value());
}

TEST(KalmanFilter, PredictMatchesIndependentStateAndProcessNoise) {
  const prediction::KalmanFilter filter(test_config());
  auto track = filter.initialize(make_detection(2.0, 3.0, 0.4), 1);
  ASSERT_TRUE(track.has_value());
  track->state(2) = 5.0;
  track->state(3) = -2.0;
  track->state(5) = 0.3;

  const prediction::TrackState prior = *track;
  constexpr double dt = 0.2;
  ASSERT_TRUE(filter.predict(*track, dt));

  EXPECT_NEAR(track->state(0), prior.state(0) + prior.state(2) * dt, kTolerance);
  EXPECT_NEAR(track->state(1), prior.state(1) + prior.state(3) * dt, kTolerance);
  EXPECT_NEAR(track->state(2), prior.state(2), kTolerance);
  EXPECT_NEAR(track->state(3), prior.state(3), kTolerance);
  EXPECT_NEAR(track->state(4), prediction::normalize_angle(prior.state(4) + prior.state(5) * dt),
              kTolerance);
  EXPECT_NEAR(track->state(5), prior.state(5), kTolerance);
  EXPECT_EQ(track->track_id, prior.track_id);
  EXPECT_EQ(track->status, prior.status);

  const Eigen::Matrix<double, 6, 6> Q =
      expected_process_noise(dt, filter.config().sigma_a_sq, filter.config().sigma_alpha_sq);
  const Eigen::Matrix<double, 6, 6> F = transition(dt);
  const Eigen::Matrix<double, 6, 6> expected =
      0.5 * ((F * prior.covariance * F.transpose() + Q) +
             (F * prior.covariance * F.transpose() + Q).transpose());
  expect_matrix_near(track->covariance, expected, kLooseTolerance);

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> q_solver(Q);
  ASSERT_EQ(q_solver.info(), Eigen::Success);
  int positive = 0;
  int zero = 0;
  for (int i = 0; i < q_solver.eigenvalues().size(); ++i) {
    const double eigenvalue = q_solver.eigenvalues()(i);
    if (std::abs(eigenvalue) <= kTolerance) {
      ++zero;
    } else if (eigenvalue > kTolerance) {
      ++positive;
    }
  }
  EXPECT_EQ(positive, 3);
  EXPECT_EQ(zero, 3);
  EXPECT_TRUE(is_symmetric(Q));
}

TEST(KalmanFilter, ZeroDtIsExactNoOp) {
  const prediction::KalmanFilter filter(test_config());
  auto track = filter.initialize(make_detection(1.0, 2.0, 3.0 * prediction::kPi), 9);
  ASSERT_TRUE(track.has_value());
  const prediction::TrackState prior = *track;
  ASSERT_TRUE(filter.predict(*track, 0.0));
  expect_unchanged(*track, prior);
}

TEST(KalmanFilter, VelocityAndTurnRateConvergeFromPositionHeadingOnly) {
  const prediction::KalmanFilter filter(test_config());
  constexpr double vx = 8.0;
  constexpr double vy = -3.0;
  constexpr double omega = 0.15;
  constexpr double dt = 0.1;
  auto track = filter.initialize(make_detection(0.0, 0.0, 0.2), 1);
  ASSERT_TRUE(track.has_value());

  double x = 0.0;
  double y = 0.0;
  double psi = 0.2;
  for (int step = 0; step < 40; ++step) {
    x += vx * dt;
    y += vy * dt;
    psi = prediction::normalize_angle(psi + omega * dt);
    ASSERT_TRUE(filter.predict(*track, dt));
    ASSERT_TRUE(filter.update(*track, make_detection(x, y, psi)));
  }
  EXPECT_NEAR(track->state(2), vx, 0.05);
  EXPECT_NEAR(track->state(3), vy, 0.05);
  EXPECT_NEAR(track->state(5), omega, 0.02);
}

TEST(KalmanFilter, HeadingWrapsAcrossBranchCutOnPredictAndUpdate) {
  const prediction::KalmanFilter filter(test_config());
  auto track = filter.initialize(make_detection(0.0, 0.0, 179.0 * prediction::kPi / 180.0), 1);
  ASSERT_TRUE(track.has_value());
  track->state(5) = 0.5;
  track->covariance.setZero();
  track->covariance.diagonal().setConstant(1e-4);
  track->covariance(5, 5) = 1.0;

  const double before = track->state(4);
  ASSERT_TRUE(filter.predict(*track, 0.2));
  EXPECT_GT(before, 0.0);
  EXPECT_LT(track->state(4), 0.0);
  EXPECT_NEAR(track->state(4), prediction::normalize_angle(before + 0.5 * 0.2), kTolerance);

  // Tight position/heading variance and a loose prior heading make the update
  // pull psi across +pi instead of leaving it short of the cut.
  track->state(4) = 179.0 * prediction::kPi / 180.0;
  track->covariance.setIdentity();
  track->covariance(4, 4) = 1.0;
  const prediction::Detection measurement =
      make_detection(0.0, 0.0, -179.0 * prediction::kPi / 180.0);
  const double innovation =
      prediction::normalize_angle(measurement.heading - track->state(4));
  EXPECT_NEAR(innovation, 2.0 * prediction::kPi / 180.0, 1e-9);
  ASSERT_TRUE(filter.update(*track, measurement));
  EXPECT_LT(track->state(4), 0.0);
  EXPECT_GT(track->state(4), -prediction::kPi);
  EXPECT_NEAR(std::abs(track->state(4)), prediction::kPi, 0.05);
}

TEST(KalmanFilter, UpdateShrinksCovarianceAndStaysSymmetricPositiveDefinite) {
  const prediction::KalmanFilter filter(test_config());
  auto track = filter.initialize(make_detection(1.0, -2.0, 0.3), 4);
  ASSERT_TRUE(track.has_value());
  const double prior_trace = track->covariance.trace();
  const double prior_determinant = track->covariance.determinant();
  ASSERT_TRUE(filter.update(*track, make_detection(1.1, -1.9, 0.32, 5.0, 2.0, 0.4)));

  EXPECT_LT(track->covariance.trace(), prior_trace);
  EXPECT_LT(track->covariance.determinant(), prior_determinant);
  EXPECT_TRUE(is_symmetric(track->covariance));
  EXPECT_NEAR(track->size.x(), 5.0, kTolerance);
  EXPECT_NEAR(track->size.y(), 2.0, kTolerance);

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(track->covariance);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

TEST(KalmanFilter, MahalanobisDistanceMatchesHandSolveAndWraps) {
  const prediction::KalmanFilter filter(test_config());
  auto track = filter.initialize(make_detection(0.0, 0.0, 179.0 * prediction::kPi / 180.0), 1);
  ASSERT_TRUE(track.has_value());
  const prediction::Detection measurement =
      make_detection(1.0, -0.5, -179.0 * prediction::kPi / 180.0);

  Eigen::Vector3d residual;
  residual << 1.0, -0.5,
      prediction::normalize_angle(measurement.heading - track->state(4));
  Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
  S(0, 0) = track->covariance(0, 0) + filter.config().sigma_pos_sq;
  S(1, 1) = track->covariance(1, 1) + filter.config().sigma_pos_sq;
  S(2, 2) = track->covariance(4, 4) + filter.config().sigma_psi_sq;
  const double expected = residual.dot(S.ldlt().solve(residual));

  const auto distance = filter.mahalanobis_distance_squared(*track, measurement);
  ASSERT_TRUE(distance.has_value());
  EXPECT_NEAR(*distance, expected, kLooseTolerance);

  prediction::Detection unwrapped = measurement;
  unwrapped.heading += prediction::kTwoPi;
  const auto wrapped = filter.mahalanobis_distance_squared(*track, unwrapped);
  ASSERT_TRUE(wrapped.has_value());
  EXPECT_NEAR(*distance, *wrapped, kTolerance);
}

TEST(KalmanFilter, RejectsBadDtAndMalformedInputWithoutMutation) {
  const prediction::KalmanFilter filter(test_config());
  auto track = filter.initialize(make_detection(3.0, 4.0, 0.5), 11);
  ASSERT_TRUE(track.has_value());
  track->status = prediction::TrackStatus::Confirmed;
  const prediction::TrackState prior = *track;

  EXPECT_FALSE(filter.predict(*track, -0.1));
  expect_unchanged(*track, prior);
  EXPECT_FALSE(filter.predict(*track, std::numeric_limits<double>::quiet_NaN()));
  expect_unchanged(*track, prior);
  EXPECT_FALSE(filter.predict(*track, std::numeric_limits<double>::infinity()));
  expect_unchanged(*track, prior);

  prediction::Detection bad = make_detection(3.0, 4.0, 0.5);
  bad.position.x() = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(filter.update(*track, bad));
  expect_unchanged(*track, prior);
  EXPECT_FALSE(filter.mahalanobis_distance_squared(*track, bad).has_value());

  bad = make_detection(3.0, 4.0, 0.5);
  bad.size.x() = -1.0;
  EXPECT_FALSE(filter.update(*track, bad));
  expect_unchanged(*track, prior);

  prediction::TrackState asymmetric = prior;
  asymmetric.covariance(0, 1) = 0.5;
  asymmetric.covariance(1, 0) = 0.0;
  prediction::TrackState before = asymmetric;
  EXPECT_FALSE(filter.predict(asymmetric, 0.1));
  expect_unchanged(asymmetric, before);
  EXPECT_FALSE(filter.mahalanobis_distance_squared(asymmetric, make_detection(3.0, 4.0, 0.5))
                   .has_value());

  prediction::TrackState indefinite = prior;
  indefinite.covariance(2, 2) = -1.0;
  before = indefinite;
  EXPECT_FALSE(filter.update(indefinite, make_detection(3.0, 4.0, 0.5)));
  expect_unchanged(indefinite, before);
}

TEST(KalmanFilter, HeaderOnlyAnchorLinksFromAnotherTranslationUnit) {
  const prediction::KalmanFilter& anchor = prediction::test_support::linkage_anchor();
  EXPECT_GT(anchor.config().sigma_a_sq, 0.0);
  const prediction::KalmanFilter local;
  EXPECT_DOUBLE_EQ(anchor.config().sigma_pos_sq, local.config().sigma_pos_sq);
}
