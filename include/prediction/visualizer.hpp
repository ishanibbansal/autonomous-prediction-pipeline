#pragma once

#include "prediction/types.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prediction {

/// Styling and camera settings for one bird's-eye-view frame.
///
/// This struct is a value. It is not validated until it is copied into a
/// BevVisualizer. A caller may build an invalid config and repair it before
/// construction; the visualizer then keeps its own validated copy. Layer toggles
/// do not skip validation of the fields that would have driven those layers.
///
/// width_pixels and height_pixels may be fractional. center_world is the world
/// point mapped to the middle of the SVG viewport, in meters. pixels_per_meter
/// is the isotropic scale. grid_spacing_meters is the world spacing of the
/// metric grid, which is anchored to world multiples rather than to the camera.
/// covariance_sigma_scale is the n in an n-sigma position contour. For a
/// full-rank 2D Gaussian a 2-sigma contour encloses approximately 86.5 percent
/// of the density, not 95 percent. ellipse_step_stride is the zero-based
/// waypoint spacing of those contours: indices 0, stride, 2 * stride, ...
/// velocity_arrow_seconds is the time used to draw a velocity vector as
/// p + tau * v. Zero is a valid omission of every arrow.
struct VisualizerConfig {
  double width_pixels{1000.0};
  double height_pixels{1000.0};
  double pixels_per_meter{20.0};
  Eigen::Vector2d center_world{Eigen::Vector2d::Zero()};
  double grid_spacing_meters{5.0};
  double covariance_sigma_scale{2.0};
  std::size_t ellipse_step_stride{5U};
  double velocity_arrow_seconds{0.5};
  bool draw_grid{true};
  bool draw_covariance_ellipses{true};
  bool draw_hud{true};
};

/// Header-only bird's-eye-view SVG renderer.
///
/// World meters (x, y) map to screen pixels (X, Y) by
///   X = W / 2 + (x - c_x) * s
///   Y = H / 2 - (y - c_y) * s
/// with s = pixels_per_meter. Positive world y is up on the page. The camera
/// may be off center and the viewport need not be square.
///
/// Tracks supply the only vehicle boxes. A prediction's embedded current_state
/// and a track's 6 x 6 covariance are not read and are not validated. When a
/// prediction has a nonempty modes vector, modes are authoritative and the
/// legacy waypoints copy is ignored, including for validation. When modes is
/// empty, a nonempty legacy waypoints vector is drawn as one forecast. That is
/// a rendering convenience, not an evaluator fallback: PredictedWaypoint has no
/// velocity or covariance. Empty collections are valid. Any mode count K is
/// accepted.
///
/// The first three mode colors are a convention for KinematicTrajectoryPredictor,
/// not inferred maneuver semantics: mode 0 is the straight hypothesis, mode 1 is
/// counterclockwise, and mode 2 is clockwise. Modes at index 3 and above receive
/// a deterministic fallback style and a numeric index.
///
/// Covariance contours are n-sigma envelopes of the 2 x 2 waypoint position
/// covariance, rotated by the major eigenvector rather than by vehicle yaw.
/// Every authoritative covariance is validated, including samples skipped by
/// ellipse_step_stride and including frames drawn with
/// draw_covariance_ellipses = false. A styling toggle cannot hide corrupt data.
///
/// Ground truth, predictions, and metrics need not share a horizon, a clock, or
/// a track id. Samples are drawn in the order supplied. The renderer does not
/// resample, score, or renormalize probabilities. Score labels join by track_id.
/// A duplicated id among tracks or among metrics records suppresses that track's
/// score line; the HUD still aggregates every metric entry.
///
/// Construction throws std::invalid_argument naming the offending config field.
/// render_frame throws std::invalid_argument for non-finite or malformed input,
/// with track, mode, and waypoint context where applicable. Non-representable
/// derived screen geometry, a failed numerical decomposition of otherwise
/// admissible finite input, or an aggregate overflow throws std::overflow_error
/// with context. The function returns a complete SVG string or throws. It never
/// emits nan or inf, never casts an unbounded grid count to an integer, and does
/// not depend on long double having greater range. Inputs are not mutated, and
/// no reference to them is retained.
class BevVisualizer {
 public:
  /// If either axis would need more than this many visible grid lines, the
  /// entire grid layer is omitted for that frame. No grid index is cast to an
  /// integer. The scene is otherwise unchanged, and the empty grid group is
  /// marked with a comment so the omission is observable.
  static constexpr std::size_t kMaxVisibleGridLines = 2048U;

  /// Relative asymmetry rejected before a 2 x 2 covariance is symmetrized.
  static constexpr double kCovarianceAsymmetryTolerance = 1e-9;

  /// Relative negative-eigenvalue floor. Values below this reject the matrix;
  /// accepted negative roundoff is clamped to zero.
  static constexpr double kEigenvalueFloorTolerance = 1e-9;

  explicit BevVisualizer(VisualizerConfig config = {}) : config_(validate(std::move(config))) {}

  const VisualizerConfig& config() const { return config_; }

  /// Render one standalone SVG frame. Empty collections are valid. The returned
  /// string is complete and does not depend on external fonts, assets, scripts,
  /// or stylesheets.
  std::string render_frame(double current_timestamp, const std::vector<TrackState>& tracks,
                           const std::vector<PredictedTrajectory>& predictions,
                           const std::vector<GroundTruthTrajectory>& ground_truths = {},
                           const std::vector<TrajectoryMetrics>& metrics = {}) const {
    if (!std::isfinite(current_timestamp)) {
      throw std::invalid_argument("current_timestamp must be finite");
    }
    validate_tracks(tracks);
    validate_predictions(predictions);
    validate_ground_truths(ground_truths);
    validate_metrics(metrics);

    Scene scene = project_scene(current_timestamp, tracks, predictions, ground_truths, metrics);

    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << format_svg(config_.width_pixels)
        << "\" height=\"" << format_svg(config_.height_pixels) << "\" viewBox=\"0 0 "
        << format_svg(config_.width_pixels) << " " << format_svg(config_.height_pixels) << "\">\n";
    out << "<defs><clipPath id=\"viewport_clip\"><rect x=\"0\" y=\"0\" width=\""
        << format_svg(config_.width_pixels) << "\" height=\"" << format_svg(config_.height_pixels)
        << "\"/></clipPath></defs>\n";
    out << "<g id=\"scene\" clip-path=\"url(#viewport_clip)\">\n";

    emit_background(out);
    emit_grid(out, scene);
    emit_ground_truth(out, scene);
    emit_uncertainty(out, scene);
    emit_predictions(out, scene);
    emit_tracks(out, scene);
    emit_labels(out, scene);
    out << "</g>\n";
    emit_hud(out, scene);
    out << "</svg>\n";

    const std::string svg = out.str();
    if (svg.find("nan") != std::string::npos || svg.find("inf") != std::string::npos ||
        svg.find("NaN") != std::string::npos || svg.find("Inf") != std::string::npos) {
      throw std::overflow_error("BevVisualizer produced a non-finite SVG number");
    }
    return svg;
  }

 private:
  struct ScreenPoint {
    double x{0.0};
    double y{0.0};
  };

  struct GridAxis {
    bool omit{false};
    bool density_exceeded{false};
    double phase_world{0.0};
    std::size_t count{0U};
    double spacing_world{0.0};
  };

  struct DrawnEllipse {
    ScreenPoint center{};
    double rx{0.0};
    double ry{0.0};
    double angle_deg{0.0};
    bool rank_one{false};
    ScreenPoint segment_start{};
    ScreenPoint segment_end{};
    std::size_t mode_index{0U};
    bool legacy{false};
  };

  struct DrawnPath {
    std::vector<ScreenPoint> points{};
    std::size_t mode_index{0U};
    bool legacy{false};
    bool ground_truth{false};
  };

  struct DrawnTrack {
    ScreenPoint center{};
    double length_px{0.0};
    double width_px{0.0};
    double yaw_deg{0.0};
    bool has_velocity_arrow{false};
    ScreenPoint velocity_tip{};
    TrackStatus status{TrackStatus::Tentative};
    std::uint64_t track_id{0U};
    bool label{true};
    bool score{false};
    double min_ade{0.0};
    double min_fde{0.0};
  };

  struct HudCounts {
    std::size_t tracks{0U};
    std::size_t confirmed{0U};
    std::size_t tentative{0U};
    std::size_t lost{0U};
    bool has_metrics{false};
    std::size_t evaluated{0U};
    double mean_min_ade{0.0};
    double mean_min_fde{0.0};
    double miss_percent{0.0};
  };

  struct Scene {
    GridAxis vertical{};
    GridAxis horizontal{};
    std::vector<DrawnPath> ground_truth{};
    std::vector<DrawnEllipse> ellipses{};
    std::vector<DrawnPath> predictions{};
    std::vector<DrawnTrack> tracks{};
    HudCounts hud{};
    double timestamp{0.0};
  };

  struct LabelBox {
    double x{0.0};
    double y{0.0};
    double width{0.0};
    double height{0.0};
  };

  static VisualizerConfig validate(VisualizerConfig config) {
    require_finite_positive(config.width_pixels, "VisualizerConfig::width_pixels");
    require_finite_positive(config.height_pixels, "VisualizerConfig::height_pixels");
    require_finite_positive(config.pixels_per_meter, "VisualizerConfig::pixels_per_meter");
    if (!config.center_world.allFinite()) {
      throw std::invalid_argument("VisualizerConfig::center_world must be finite");
    }
    require_finite_positive(config.grid_spacing_meters, "VisualizerConfig::grid_spacing_meters");
    require_finite_positive(config.covariance_sigma_scale, "VisualizerConfig::covariance_sigma_scale");
    if (config.ellipse_step_stride < 1U) {
      throw std::invalid_argument("VisualizerConfig::ellipse_step_stride must be >= 1");
    }
    if (!std::isfinite(config.velocity_arrow_seconds) || config.velocity_arrow_seconds < 0.0) {
      throw std::invalid_argument(
          "VisualizerConfig::velocity_arrow_seconds must be finite and nonnegative");
    }
    return config;
  }

  static void require_finite_positive(double value, const char* name) {
    if (!std::isfinite(value) || !(value > 0.0)) {
      throw std::invalid_argument(std::string(name) + " must be finite and strictly positive");
    }
  }

  static void require_finite(double value, const std::string& message) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(message);
    }
  }

  static std::string track_where(std::uint64_t track_id, std::size_t index) {
    return "track_id " + std::to_string(track_id) + ", track index " + std::to_string(index);
  }

  static bool known_status(TrackStatus status) {
    return status == TrackStatus::Tentative || status == TrackStatus::Confirmed ||
           status == TrackStatus::Lost;
  }

  static void validate_tracks(const std::vector<TrackState>& tracks) {
    for (std::size_t index = 0; index < tracks.size(); ++index) {
      const TrackState& track = tracks[index];
      const std::string where = track_where(track.track_id, index);
      if (!known_status(track.status)) {
        throw std::invalid_argument("TrackState::status is not a known TrackStatus (" + where + ")");
      }
      if (!std::isfinite(track.size.x()) || !(track.size.x() > 0.0)) {
        throw std::invalid_argument(
            "TrackState::size.x must be finite and strictly positive (" + where + ")");
      }
      if (!std::isfinite(track.size.y()) || !(track.size.y() > 0.0)) {
        throw std::invalid_argument(
            "TrackState::size.y must be finite and strictly positive (" + where + ")");
      }
      if (!track.state.allFinite()) {
        throw std::invalid_argument("TrackState::state must be finite (" + where + ")");
      }
    }
  }

  static void validate_covariance(const Eigen::Matrix2d& covariance, const std::string& where) {
    if (!covariance.allFinite()) {
      throw std::invalid_argument("position_covariance must be finite (" + where + ")");
    }
    const double a = covariance(0, 1);
    const double b = covariance(1, 0);
    const double scale = 1.0 + std::max(std::abs(a), std::abs(b));
    const double tolerance = kCovarianceAsymmetryTolerance * scale;
    const double difference = a - b;
    if (!std::isfinite(difference) || std::abs(difference) > tolerance) {
      throw std::invalid_argument("position_covariance is not symmetric (" + where + ")");
    }
    // Eigenvalue admissibility is part of input validation, not a draw toggle.
    // A skipped stride sample and a frame with ellipses disabled still reject
    // an indefinite matrix. The returned geometry is discarded here.
    (void)solve_ellipse(covariance, 1.0, where);
  }

  static void validate_mode_points(const std::vector<TrajectoryPoint>& waypoints, const std::string& where) {
    for (std::size_t index = 0; index < waypoints.size(); ++index) {
      const TrajectoryPoint& point = waypoints[index];
      const std::string sample = where + ", waypoint " + std::to_string(index);
      require_finite(point.timestamp, "TrajectoryPoint::timestamp must be finite (" + sample + ")");
      require_finite(point.heading, "TrajectoryPoint::heading must be finite (" + sample + ")");
      if (!point.position.allFinite()) {
        throw std::invalid_argument("TrajectoryPoint::position must be finite (" + sample + ")");
      }
      if (!point.velocity.allFinite()) {
        throw std::invalid_argument("TrajectoryPoint::velocity must be finite (" + sample + ")");
      }
      validate_covariance(point.position_covariance, sample);
    }
  }

  static void validate_predictions(const std::vector<PredictedTrajectory>& predictions) {
    for (std::size_t index = 0; index < predictions.size(); ++index) {
      const PredictedTrajectory& prediction = predictions[index];
      const std::string where =
          "track_id " + std::to_string(prediction.track_id) + ", prediction index " + std::to_string(index);
      if (!prediction.modes.empty()) {
        for (std::size_t mode_index = 0; mode_index < prediction.modes.size(); ++mode_index) {
          const TrajectoryMode& mode = prediction.modes[mode_index];
          const std::string mode_where = where + ", mode " + std::to_string(mode_index);
          require_finite(mode.probability, "TrajectoryMode::probability must be finite (" + mode_where + ")");
          validate_mode_points(mode.waypoints, mode_where);
        }
        continue;
      }
      for (std::size_t waypoint_index = 0; waypoint_index < prediction.waypoints.size(); ++waypoint_index) {
        const PredictedWaypoint& waypoint = prediction.waypoints[waypoint_index];
        const std::string sample = where + ", legacy waypoint " + std::to_string(waypoint_index);
        require_finite(waypoint.timestamp, "PredictedWaypoint::timestamp must be finite (" + sample + ")");
        require_finite(waypoint.heading, "PredictedWaypoint::heading must be finite (" + sample + ")");
        if (!waypoint.position.allFinite()) {
          throw std::invalid_argument("PredictedWaypoint::position must be finite (" + sample + ")");
        }
      }
    }
  }

  static void validate_ground_truths(const std::vector<GroundTruthTrajectory>& ground_truths) {
    for (std::size_t index = 0; index < ground_truths.size(); ++index) {
      const GroundTruthTrajectory& truth = ground_truths[index];
      for (std::size_t waypoint_index = 0; waypoint_index < truth.waypoints.size(); ++waypoint_index) {
        const GroundTruthWaypoint& waypoint = truth.waypoints[waypoint_index];
        const std::string sample = "track_id " + std::to_string(truth.track_id) +
                                   ", ground-truth index " + std::to_string(index) + ", waypoint " +
                                   std::to_string(waypoint_index);
        require_finite(waypoint.timestamp, "GroundTruthWaypoint::timestamp must be finite (" + sample + ")");
        if (!waypoint.position.allFinite()) {
          throw std::invalid_argument("GroundTruthWaypoint::position must be finite (" + sample + ")");
        }
      }
    }
  }

  static void validate_metrics(const std::vector<TrajectoryMetrics>& metrics) {
    for (std::size_t index = 0; index < metrics.size(); ++index) {
      const TrajectoryMetrics& entry = metrics[index];
      const std::string where =
          "track_id " + std::to_string(entry.track_id) + ", metrics index " + std::to_string(index);
      if (!std::isfinite(entry.min_ade) || entry.min_ade < 0.0) {
        throw std::invalid_argument("TrajectoryMetrics::min_ade must be finite and nonnegative (" + where +
                                    ")");
      }
      if (!std::isfinite(entry.min_fde) || entry.min_fde < 0.0) {
        throw std::invalid_argument("TrajectoryMetrics::min_fde must be finite and nonnegative (" + where +
                                    ")");
      }
    }
  }

  ScreenPoint project(double world_x, double world_y, const std::string& where) const {
    const double dx = world_x - config_.center_world.x();
    const double dy = world_y - config_.center_world.y();
    if (!std::isfinite(dx) || !std::isfinite(dy)) {
      throw std::overflow_error("world offset is non-representable (" + where + ")");
    }
    const double sx = dx * config_.pixels_per_meter;
    const double sy = dy * config_.pixels_per_meter;
    if (!std::isfinite(sx) || !std::isfinite(sy)) {
      throw std::overflow_error("projected offset is non-representable (" + where + ")");
    }
    const double x = config_.width_pixels * 0.5 + sx;
    const double y = config_.height_pixels * 0.5 - sy;
    if (!std::isfinite(x) || !std::isfinite(y)) {
      throw std::overflow_error("screen coordinate is non-representable (" + where + ")");
    }
    return ScreenPoint{x, y};
  }

  double pixels(double meters, const std::string& where) const {
    if (!std::isfinite(meters)) {
      throw std::overflow_error("metric length is non-representable (" + where + ")");
    }
    const double px = meters * config_.pixels_per_meter;
    if (!std::isfinite(px)) {
      throw std::overflow_error("pixel length is non-representable (" + where + ")");
    }
    return px;
  }

  static double checked_sum(double lhs, double rhs, const std::string& where) {
    const double sum = lhs + rhs;
    if (!std::isfinite(sum)) {
      throw std::overflow_error("numeric sum overflow (" + where + ")");
    }
    return sum;
  }

  /// Average of two finite values. Splits the sum so an accepted covariance
  /// off-diagonal pair cannot overflow while forming (a + b) / 2.
  static double symmetrized_off_diagonal(double a, double b, const std::string& where) {
    const double half_a = a * 0.5;
    const double half_b = b * 0.5;
    if (!std::isfinite(half_a) || !std::isfinite(half_b)) {
      throw std::overflow_error("covariance symmetrization overflow (" + where + ")");
    }
    return checked_sum(half_a, half_b, where + " covariance symmetrization");
  }

  static double checked_scale(double value, double scale, const std::string& where) {
    const double product = value * scale;
    if (!std::isfinite(product)) {
      throw std::overflow_error("numeric scale overflow (" + where + ")");
    }
    return product;
  }

  GridAxis plan_grid_axis(double origin, double extent, bool vertical) const {
    GridAxis axis;
    axis.spacing_world = config_.grid_spacing_meters;
    const double half = extent * 0.5 / config_.pixels_per_meter;
    if (!std::isfinite(half) || !std::isfinite(origin)) {
      throw std::overflow_error("grid extent is non-representable");
    }
    const double first = origin - half;
    const double last = origin + half;
    if (!std::isfinite(first) || !std::isfinite(last)) {
      throw std::overflow_error("grid window is non-representable");
    }
    // A span that cannot be counted in a bounded integer is omitted rather than
    // walked. The omitted axis is not an error: extreme cameras still render.
    const double span = last - first;
    if (!std::isfinite(span)) {
      axis.omit = true;
      return axis;
    }
    const double estimated = span / config_.grid_spacing_meters;
    if (!std::isfinite(estimated) || estimated > static_cast<double>(kMaxVisibleGridLines) + 2.0) {
      axis.omit = true;
      axis.density_exceeded = true;
      return axis;
    }

    const double first_index = std::ceil(first / config_.grid_spacing_meters);
    const double last_index = std::floor(last / config_.grid_spacing_meters);
    if (!std::isfinite(first_index) || !std::isfinite(last_index)) {
      axis.omit = true;
      axis.density_exceeded = true;
      return axis;
    }
    if (last_index < first_index) {
      axis.count = 0U;
      axis.phase_world = first_index * config_.grid_spacing_meters;
      return axis;
    }
    const double count = last_index - first_index;
    if (!std::isfinite(count) || count >= static_cast<double>(kMaxVisibleGridLines)) {
      axis.omit = true;
      axis.density_exceeded = true;
      return axis;
    }
    const double phase = first_index * config_.grid_spacing_meters;
    if (!std::isfinite(phase)) {
      axis.omit = true;
      axis.density_exceeded = true;
      return axis;
    }
    axis.phase_world = phase;
    // count is in [0, kMaxVisibleGridLines), so count + 1 fits in the cap.
    axis.count = static_cast<std::size_t>(count) + 1U;
    (void)vertical;
    return axis;
  }

  struct EllipseGeometry {
    double major{0.0};
    double minor{0.0};
    double angle_deg{0.0};
    bool omit{false};
    bool rank_one{false};
  };

  static EllipseGeometry solve_ellipse(const Eigen::Matrix2d& covariance, double sigma, const std::string& where) {
    const double a = covariance(0, 1);
    const double b = covariance(1, 0);
    const double off = symmetrized_off_diagonal(a, b, where);
    Eigen::Matrix2d symmetric;
    symmetric << covariance(0, 0), off, off, covariance(1, 1);
    if (!symmetric.allFinite()) {
      throw std::overflow_error("symmetrized covariance is non-representable (" + where + ")");
    }

    const double norm = std::max(std::abs(symmetric(0, 0)),
                                 std::max(std::abs(symmetric(1, 1)), std::abs(symmetric(0, 1))));
    const double scale = std::isfinite(norm) && norm > 1.0 ? norm : 1.0;
    Eigen::Matrix2d normalized = symmetric / scale;
    if (!normalized.allFinite()) {
      throw std::overflow_error("normalized covariance is non-representable (" + where + ")");
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(normalized);
    if (solver.info() != Eigen::Success) {
      throw std::overflow_error("covariance eigensolve failed (" + where + ")");
    }
    const Eigen::Vector2d normalized_values = solver.eigenvalues();
    if (!normalized_values.allFinite()) {
      throw std::overflow_error("covariance eigenvalues are non-finite (" + where + ")");
    }
    // Eigen returns ascending eigenvalues. The last one is the major axis.
    const double lambda_min_n = normalized_values(0);
    const double lambda_max_n = normalized_values(1);
    const double lambda_min = checked_scale(lambda_min_n, scale, where + " eigenvalue");
    const double lambda_max = checked_scale(lambda_max_n, scale, where + " eigenvalue");
    const double floor = -kEigenvalueFloorTolerance * (1.0 + std::max(0.0, lambda_max));
    if (!std::isfinite(floor) || lambda_min < floor) {
      throw std::invalid_argument("position_covariance is indefinite (" + where + ")");
    }
    const double clamped_min = lambda_min < 0.0 ? 0.0 : lambda_min;
    const double clamped_max = lambda_max < 0.0 ? 0.0 : lambda_max;

    EllipseGeometry geometry;
    if (!(clamped_max > 0.0)) {
      geometry.omit = true;
      return geometry;
    }
    geometry.rank_one = !(clamped_min > 0.0);
    const double root_major = std::sqrt(clamped_max);
    const double root_minor = std::sqrt(clamped_min);
    if (!std::isfinite(root_major) || !std::isfinite(root_minor)) {
      throw std::overflow_error("covariance semi-axis is non-representable (" + where + ")");
    }
    geometry.major = checked_scale(sigma, root_major, where + " semi-axis");
    geometry.minor = checked_scale(sigma, root_minor, where + " semi-axis");

    const Eigen::Vector2d eigenvector = solver.eigenvectors().col(1);
    if (!eigenvector.allFinite()) {
      throw std::overflow_error("covariance eigenvector is non-finite (" + where + ")");
    }
    const double vx = eigenvector.x();
    const double vy = eigenvector.y();
    if (clamped_min == clamped_max) {
      geometry.angle_deg = 0.0;
    } else {
      // Screen y is inverted, so the displayed slope of (vx, vy) is -vy / vx.
      // A 180 degree flip describes the same ellipse; fold into (-90, 90].
      double degrees = std::atan2(-vy, vx) * (180.0 / kPi);
      if (!std::isfinite(degrees)) {
        throw std::overflow_error("covariance orientation is non-representable (" + where + ")");
      }
      if (degrees <= -90.0) {
        degrees += 180.0;
      } else if (degrees > 90.0) {
        degrees -= 180.0;
      }
      geometry.angle_deg = degrees;
    }
    return geometry;
  }

  static bool index_is_stride(std::size_t index, std::size_t stride) {
    if (stride == 0U) {
      return false;
    }
    return index % stride == 0U;
  }

  Scene project_scene(double current_timestamp, const std::vector<TrackState>& tracks,
                      const std::vector<PredictedTrajectory>& predictions,
                      const std::vector<GroundTruthTrajectory>& ground_truths,
                      const std::vector<TrajectoryMetrics>& metrics) const {
    Scene scene;
    scene.timestamp = current_timestamp;
    scene.vertical = plan_grid_axis(config_.center_world.x(), config_.width_pixels, true);
    scene.horizontal = plan_grid_axis(config_.center_world.y(), config_.height_pixels, false);
    project_ground_truth(ground_truths, scene);
    project_predictions(predictions, scene);
    project_tracks(tracks, metrics, scene);
    project_hud(current_timestamp, tracks, metrics, scene);
    return scene;
  }

  void project_ground_truth(const std::vector<GroundTruthTrajectory>& ground_truths, Scene& scene) const {
    scene.ground_truth.reserve(ground_truths.size());
    for (std::size_t index = 0; index < ground_truths.size(); ++index) {
      const GroundTruthTrajectory& truth = ground_truths[index];
      DrawnPath path;
      path.ground_truth = true;
      path.points.reserve(truth.waypoints.size());
      for (std::size_t waypoint_index = 0; waypoint_index < truth.waypoints.size(); ++waypoint_index) {
        const std::string where = "track_id " + std::to_string(truth.track_id) + ", ground-truth index " +
                                  std::to_string(index) + ", waypoint " + std::to_string(waypoint_index);
        path.points.push_back(project(truth.waypoints[waypoint_index].position.x(),
                                      truth.waypoints[waypoint_index].position.y(), where));
      }
      scene.ground_truth.push_back(std::move(path));
    }
  }

  void project_predictions(const std::vector<PredictedTrajectory>& predictions, Scene& scene) const {
    for (std::size_t index = 0; index < predictions.size(); ++index) {
      const PredictedTrajectory& prediction = predictions[index];
      if (!prediction.modes.empty()) {
        for (std::size_t mode_index = 0; mode_index < prediction.modes.size(); ++mode_index) {
          project_mode(prediction, index, mode_index, scene);
        }
        continue;
      }
      if (!prediction.waypoints.empty()) {
        project_legacy(prediction, index, scene);
      }
    }
  }

  void project_mode(const PredictedTrajectory& prediction, std::size_t prediction_index, std::size_t mode_index,
                    Scene& scene) const {
    const TrajectoryMode& mode = prediction.modes[mode_index];
    DrawnPath path;
    path.mode_index = mode_index;
    path.points.reserve(mode.waypoints.size());
    for (std::size_t waypoint_index = 0; waypoint_index < mode.waypoints.size(); ++waypoint_index) {
      const TrajectoryPoint& point = mode.waypoints[waypoint_index];
      const std::string where = "track_id " + std::to_string(prediction.track_id) + ", prediction index " +
                                std::to_string(prediction_index) + ", mode " + std::to_string(mode_index) +
                                ", waypoint " + std::to_string(waypoint_index);
      const ScreenPoint center = project(point.position.x(), point.position.y(), where);
      path.points.push_back(center);
      if (!config_.draw_covariance_ellipses || !index_is_stride(waypoint_index, config_.ellipse_step_stride)) {
        continue;
      }
      const EllipseGeometry geometry =
          solve_ellipse(point.position_covariance, config_.covariance_sigma_scale, where);
      if (geometry.omit) {
        continue;
      }
      DrawnEllipse ellipse;
      ellipse.center = center;
      ellipse.mode_index = mode_index;
      ellipse.angle_deg = geometry.angle_deg;
      ellipse.rx = pixels(geometry.major, where + " major axis");
      ellipse.ry = pixels(geometry.minor, where + " minor axis");
      ellipse.rank_one = geometry.rank_one;
      if (ellipse.rank_one) {
        const double theta = geometry.angle_deg * kPi / 180.0;
        const double along_x = ellipse.rx * std::cos(theta);
        const double along_y = ellipse.rx * std::sin(theta);
        if (!std::isfinite(along_x) || !std::isfinite(along_y)) {
          throw std::overflow_error("rank-one contour is non-representable (" + where + ")");
        }
        ellipse.segment_start =
            ScreenPoint{checked_sum(center.x, -along_x, where + " rank-one tip"),
                        checked_sum(center.y, -along_y, where + " rank-one tip")};
        ellipse.segment_end =
            ScreenPoint{checked_sum(center.x, along_x, where + " rank-one tip"),
                        checked_sum(center.y, along_y, where + " rank-one tip")};
      }
      scene.ellipses.push_back(ellipse);
    }
    scene.predictions.push_back(std::move(path));
  }

  void project_legacy(const PredictedTrajectory& prediction, std::size_t prediction_index, Scene& scene) const {
    DrawnPath path;
    path.legacy = true;
    path.points.reserve(prediction.waypoints.size());
    for (std::size_t waypoint_index = 0; waypoint_index < prediction.waypoints.size(); ++waypoint_index) {
      const PredictedWaypoint& point = prediction.waypoints[waypoint_index];
      const std::string where = "track_id " + std::to_string(prediction.track_id) + ", prediction index " +
                                std::to_string(prediction_index) + ", legacy waypoint " +
                                std::to_string(waypoint_index);
      path.points.push_back(project(point.position.x(), point.position.y(), where));
    }
    scene.predictions.push_back(std::move(path));
  }

  void project_tracks(const std::vector<TrackState>& tracks, const std::vector<TrajectoryMetrics>& metrics,
                      Scene& scene) const {
    scene.tracks.reserve(tracks.size());
    for (std::size_t index = 0; index < tracks.size(); ++index) {
      const TrackState& track = tracks[index];
      const std::string where = track_where(track.track_id, index);
      DrawnTrack drawn;
      drawn.center = project(track.state(0), track.state(1), where);
      drawn.length_px = pixels(track.size.x(), where + " length");
      drawn.width_px = pixels(track.size.y(), where + " width");
      drawn.yaw_deg = -track.state(4) * (180.0 / kPi);
      if (!std::isfinite(drawn.yaw_deg)) {
        throw std::overflow_error("track yaw is non-representable (" + where + ")");
      }
      drawn.status = track.status;
      drawn.track_id = track.track_id;
      const double vx = track.state(2);
      const double vy = track.state(3);
      if (config_.velocity_arrow_seconds > 0.0 && (vx != 0.0 || vy != 0.0)) {
        const double tip_x = checked_sum(track.state(0), checked_scale(config_.velocity_arrow_seconds, vx, where),
                                        where + " velocity tip");
        const double tip_y = checked_sum(track.state(1), checked_scale(config_.velocity_arrow_seconds, vy, where),
                                        where + " velocity tip");
        drawn.velocity_tip = project(tip_x, tip_y, where + " velocity tip");
        drawn.has_velocity_arrow =
            drawn.velocity_tip.x != drawn.center.x || drawn.velocity_tip.y != drawn.center.y;
      }
      const TrajectoryMetrics* score = unique_metric(track.track_id, tracks, metrics);
      if (score != nullptr) {
        drawn.score = true;
        drawn.min_ade = score->min_ade;
        drawn.min_fde = score->min_fde;
      }
      scene.tracks.push_back(drawn);
    }
  }

  static const TrajectoryMetrics* unique_metric(std::uint64_t track_id, const std::vector<TrackState>& tracks,
                                                const std::vector<TrajectoryMetrics>& metrics) {
    std::size_t track_hits = 0U;
    for (const TrackState& track : tracks) {
      if (track.track_id == track_id) {
        ++track_hits;
      }
    }
    if (track_hits != 1U) {
      return nullptr;
    }
    const TrajectoryMetrics* found = nullptr;
    for (const TrajectoryMetrics& entry : metrics) {
      if (entry.track_id != track_id) {
        continue;
      }
      if (found != nullptr) {
        return nullptr;
      }
      found = &entry;
    }
    return found;
  }

  static void project_hud(double current_timestamp, const std::vector<TrackState>& tracks,
                          const std::vector<TrajectoryMetrics>& metrics, Scene& scene) {
    scene.hud.tracks = tracks.size();
    scene.timestamp = current_timestamp;
    for (const TrackState& track : tracks) {
      if (track.status == TrackStatus::Confirmed) {
        ++scene.hud.confirmed;
      } else if (track.status == TrackStatus::Tentative) {
        ++scene.hud.tentative;
      } else if (track.status == TrackStatus::Lost) {
        ++scene.hud.lost;
      }
    }
    if (metrics.empty()) {
      return;
    }
    scene.hud.has_metrics = true;
    scene.hud.evaluated = metrics.size();
    double ade_sum = 0.0;
    double fde_sum = 0.0;
    std::size_t misses = 0U;
    for (std::size_t index = 0; index < metrics.size(); ++index) {
      const std::string where = "metrics index " + std::to_string(index);
      ade_sum = checked_sum(ade_sum, metrics[index].min_ade, where + " min_ade");
      fde_sum = checked_sum(fde_sum, metrics[index].min_fde, where + " min_fde");
      if (metrics[index].is_miss) {
        ++misses;
      }
    }
    const double count = static_cast<double>(metrics.size());
    scene.hud.mean_min_ade = ade_sum / count;
    scene.hud.mean_min_fde = fde_sum / count;
    scene.hud.miss_percent = (static_cast<double>(misses) / count) * 100.0;
    if (!std::isfinite(scene.hud.mean_min_ade) || !std::isfinite(scene.hud.mean_min_fde) ||
        !std::isfinite(scene.hud.miss_percent)) {
      throw std::overflow_error("HUD aggregate is non-finite");
    }
  }

  static std::string format_svg(double value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << value;
    return out.str();
  }

  static std::string format_hud(double value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
  }

  static std::string format_id(std::uint64_t track_id) { return std::to_string(track_id); }

  static const char* status_name(TrackStatus status) {
    if (status == TrackStatus::Confirmed) {
      return "Confirmed";
    }
    if (status == TrackStatus::Lost) {
      return "Lost";
    }
    return "Tentative";
  }

  static const char* status_color(TrackStatus status) {
    if (status == TrackStatus::Confirmed) {
      return "#34D399";
    }
    if (status == TrackStatus::Lost) {
      return "#FB7185";
    }
    return "#FBBF24";
  }

  static const char* status_dash(TrackStatus status) {
    if (status == TrackStatus::Tentative) {
      return "4 3";
    }
    if (status == TrackStatus::Lost) {
      return "10 5";
    }
    return nullptr;
  }

  struct ModeStyle {
    const char* color;
    const char* dash;
    std::string label;
  };

  static ModeStyle mode_style(std::size_t mode_index, bool legacy) {
    ModeStyle style;
    if (legacy) {
      style.color = "#94A3B8";
      style.dash = "2 2";
      style.label = "legacy";
      return style;
    }
    // First-three colors match KinematicTrajectoryPredictor mode order. They
    // are a documented convention, not inferred maneuver semantics.
    if (mode_index == 0U) {
      style.color = "#60A5FA";
      style.dash = nullptr;
      style.label = "0";
      return style;
    }
    if (mode_index == 1U) {
      style.color = "#C084FC";
      style.dash = "6 3";
      style.label = "1";
      return style;
    }
    if (mode_index == 2U) {
      style.color = "#22D3EE";
      style.dash = "2 2";
      style.label = "2";
      return style;
    }
    static constexpr const char* kFallbackColors[] = {"#F472B6", "#A3E635", "#F59E0B", "#818CF8",
                                                       "#FB923C", "#2DD4BF", "#E879F9", "#FACC15"};
    style.color = kFallbackColors[mode_index % 8U];
    style.dash = mode_index % 2U == 0U ? "8 2 2 2" : "1 3";
    style.label = std::to_string(mode_index);
    return style;
  }

  void emit_background(std::ostream& out) const {
    out << "<g id=\"background\">\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << format_svg(config_.width_pixels) << "\" height=\""
        << format_svg(config_.height_pixels) << "\" fill=\"#0B1220\"/>\n";
    out << "</g>\n";
  }

  void emit_grid(std::ostream& out, const Scene& scene) const {
    out << "<g id=\"grid\" fill=\"none\" stroke=\"#263449\" stroke-width=\"1\">\n";
    // Either axis over the cap omits the whole grid. A non-finite window does too.
    if (!config_.draw_grid || scene.vertical.omit || scene.horizontal.omit) {
      if (config_.draw_grid &&
          (scene.vertical.density_exceeded || scene.horizontal.density_exceeded)) {
        out << "<!-- grid omitted: visible line count exceeds " << kMaxVisibleGridLines << " -->\n";
      }
      out << "</g>\n";
      return;
    }
    for (std::size_t index = 0; index < scene.vertical.count; ++index) {
      const double world_x =
          scene.vertical.phase_world + static_cast<double>(index) * scene.vertical.spacing_world;
      if (!std::isfinite(world_x)) {
        throw std::overflow_error("grid x coordinate is non-representable");
      }
      const ScreenPoint point = project(world_x, config_.center_world.y(), "grid vertical");
      out << "<line x1=\"" << format_svg(point.x) << "\" y1=\"0\" x2=\"" << format_svg(point.x) << "\" y2=\""
          << format_svg(config_.height_pixels) << "\"/>\n";
    }
    for (std::size_t index = 0; index < scene.horizontal.count; ++index) {
      const double world_y =
          scene.horizontal.phase_world + static_cast<double>(index) * scene.horizontal.spacing_world;
      if (!std::isfinite(world_y)) {
        throw std::overflow_error("grid y coordinate is non-representable");
      }
      const ScreenPoint point = project(config_.center_world.x(), world_y, "grid horizontal");
      out << "<line x1=\"0\" y1=\"" << format_svg(point.y) << "\" x2=\"" << format_svg(config_.width_pixels)
          << "\" y2=\"" << format_svg(point.y) << "\"/>\n";
    }
    out << "</g>\n";
  }

  static void emit_polyline(std::ostream& out, const std::vector<ScreenPoint>& points, const char* color,
                            const char* dash, double width, double opacity, const char* data_mode) {
    if (points.empty()) {
      return;
    }
    if (points.size() == 1U) {
      out << "<circle cx=\"" << format_svg(points[0].x) << "\" cy=\"" << format_svg(points[0].y) << "\" r=\""
          << format_svg(std::max(width, 1.5)) << "\" fill=\"" << color << "\" fill-opacity=\""
          << format_svg(opacity) << "\"";
      if (data_mode != nullptr) {
        out << " data-mode=\"" << data_mode << "\"";
      }
      out << "/>\n";
      return;
    }
    out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"" << format_svg(width)
        << "\" stroke-opacity=\"" << opacity << "\" stroke-linecap=\"round\" stroke-linejoin=\"round\"";
    if (data_mode != nullptr) {
      out << " data-mode=\"" << data_mode << "\"";
    }
    if (dash != nullptr) {
      out << " stroke-dasharray=\"" << dash << "\"";
    }
    out << " points=\"";
    for (std::size_t index = 0; index < points.size(); ++index) {
      if (index != 0U) {
        out << " ";
      }
      out << format_svg(points[index].x) << "," << format_svg(points[index].y);
    }
    out << "\"/>\n";
  }

  static void emit_ground_truth(std::ostream& out, const Scene& scene) {
    out << "<g id=\"ground_truth\">\n";
    for (const DrawnPath& path : scene.ground_truth) {
      emit_polyline(out, path.points, "#CBD5E1", "6 4", 1.5, 0.95, nullptr);
    }
    out << "</g>\n";
  }

  static void emit_uncertainty(std::ostream& out, const Scene& scene) {
    out << "<g id=\"uncertainty\" fill-opacity=\"0.18\" stroke-opacity=\"0.7\">\n";
    for (const DrawnEllipse& ellipse : scene.ellipses) {
      const ModeStyle style = mode_style(ellipse.mode_index, ellipse.legacy);
      if (ellipse.rank_one) {
        out << "<line data-mode=\"" << style.label << "\" x1=\"" << format_svg(ellipse.segment_start.x)
            << "\" y1=\"" << format_svg(ellipse.segment_start.y) << "\" x2=\""
            << format_svg(ellipse.segment_end.x) << "\" y2=\"" << format_svg(ellipse.segment_end.y)
            << "\" stroke=\"" << style.color << "\" stroke-width=\"1.5\"/>\n";
        continue;
      }
      out << "<ellipse data-mode=\"" << style.label << "\" cx=\"" << format_svg(ellipse.center.x)
          << "\" cy=\"" << format_svg(ellipse.center.y) << "\" rx=\"" << format_svg(ellipse.rx)
          << "\" ry=\"" << format_svg(ellipse.ry) << "\" transform=\"rotate(" << format_svg(ellipse.angle_deg)
          << " " << format_svg(ellipse.center.x) << " " << format_svg(ellipse.center.y) << ")\" fill=\""
          << style.color << "\" stroke=\"" << style.color << "\" stroke-width=\"1\"/>\n";
    }
    out << "</g>\n";
  }

  static void emit_predictions(std::ostream& out, const Scene& scene) {
    out << "<g id=\"predictions\">\n";
    for (const DrawnPath& path : scene.predictions) {
      const ModeStyle style = mode_style(path.mode_index, path.legacy);
      emit_polyline(out, path.points, style.color, style.dash, 2.0, 0.95, style.label.c_str());
    }
    out << "</g>\n";
  }

  static void emit_tracks(std::ostream& out, const Scene& scene) {
    out << "<g id=\"tracks\">\n";
    for (const DrawnTrack& track : scene.tracks) {
      const char* color = status_color(track.status);
      const char* dash = status_dash(track.status);
      out << "<g transform=\"translate(" << format_svg(track.center.x) << " " << format_svg(track.center.y)
          << ") rotate(" << format_svg(track.yaw_deg) << ")\">\n";
      out << "<rect x=\"" << format_svg(-0.5 * track.length_px) << "\" y=\""
          << format_svg(-0.5 * track.width_px) << "\" width=\"" << format_svg(track.length_px) << "\" height=\""
          << format_svg(track.width_px) << "\" fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2\"";
      if (dash != nullptr) {
        out << " stroke-dasharray=\"" << dash << "\"";
      }
      out << "/>\n";
      out << "<line x1=\"0\" y1=\"0\" x2=\"" << format_svg(0.5 * track.length_px)
          << "\" y2=\"0\" stroke=\"" << color << "\" stroke-width=\"2\"/>\n";
      out << "</g>\n";
      if (track.has_velocity_arrow) {
        out << "<line x1=\"" << format_svg(track.center.x) << "\" y1=\"" << format_svg(track.center.y)
            << "\" x2=\"" << format_svg(track.velocity_tip.x) << "\" y2=\""
            << format_svg(track.velocity_tip.y) << "\" stroke=\"" << color
            << "\" stroke-width=\"1.5\" marker-end=\"none\"/>\n";
        out << "<circle cx=\"" << format_svg(track.velocity_tip.x) << "\" cy=\""
            << format_svg(track.velocity_tip.y) << "\" r=\"2.5\" fill=\"" << color << "\"/>\n";
      }
    }
    out << "</g>\n";
  }

  void emit_labels(std::ostream& out, const Scene& scene) const {
    out << "<g id=\"labels\" font-family=\"monospace\" font-size=\"12\" fill=\"#E6EDF3\">\n";
    std::vector<LabelBox> occupied;
    if (config_.draw_hud) {
      occupied.push_back(hud_box(scene));
    }
    for (const DrawnTrack& track : scene.tracks) {
      if (!track.label) {
        continue;
      }
      place_label(out, track, occupied);
    }
    out << "</g>\n";
  }

  LabelBox hud_box(const Scene& scene) const {
    const double lines = scene.hud.has_metrics ? 4.0 : 2.0;
    const double width = std::min(280.0, config_.width_pixels);
    const double height = std::min(18.0 * lines + 16.0, config_.height_pixels);
    return LabelBox{8.0, 8.0, width, height};
  }

  void place_label(std::ostream& out, const DrawnTrack& track, std::vector<LabelBox>& occupied) const {
    const std::string primary = "#" + format_id(track.track_id) + " " + status_name(track.status);
    const std::string score = format_hud(track.min_ade) + " / " + format_hud(track.min_fde);
    const double text_width = static_cast<double>(primary.size()) * 7.2 + 8.0;
    const double score_width = static_cast<double>(score.size()) * 7.2 + 8.0;
    const double full_width = std::max(text_width, track.score ? score_width : text_width);
    const double line_height = 14.0;
    const double full_height = track.score ? 2.0 * line_height + 6.0 : line_height + 4.0;
    const double short_height = line_height + 4.0;

    const double margin = 6.0;
    const double candidates[8][2] = {
        {track.length_px * 0.5 + margin, -short_height},
        {track.length_px * 0.5 + margin, track.width_px * 0.5 + margin},
        {-full_width - margin, -short_height},
        {track.length_px * 0.5 + margin, -track.width_px * 0.5 - full_height},
        {-full_width * 0.5, -track.width_px * 0.5 - full_height - margin},
        {-full_width * 0.5, track.width_px * 0.5 + margin},
        {-full_width - margin, track.width_px * 0.5 + margin},
        {track.length_px * 0.5 + margin, -full_height - margin},
    };

    auto fits = [&](double x, double y, double width, double height) {
      if (x < 0.0 || y < 0.0 || x + width > config_.width_pixels || y + height > config_.height_pixels) {
        return false;
      }
      for (const LabelBox& box : occupied) {
        const bool separated = x + width < box.x || box.x + box.width < x || y + height < box.y ||
                               box.y + box.height < y;
        if (!separated) {
          return false;
        }
      }
      return true;
    };

    double chosen_x = 0.0;
    double chosen_y = 0.0;
    double chosen_width = 0.0;
    double chosen_height = 0.0;
    bool include_score = false;
    bool placed = false;
    for (const auto& offset : candidates) {
      const double x = track.center.x + offset[0];
      const double y = track.center.y + offset[1];
      if (track.score && fits(x, y, full_width, full_height)) {
        chosen_x = x;
        chosen_y = y;
        chosen_width = full_width;
        chosen_height = full_height;
        include_score = true;
        placed = true;
        break;
      }
      if (fits(x, y, text_width, short_height)) {
        chosen_x = x;
        chosen_y = y;
        chosen_width = text_width;
        chosen_height = short_height;
        include_score = false;
        placed = true;
        break;
      }
    }
    if (!placed) {
      return;
    }
    occupied.push_back(LabelBox{chosen_x, chosen_y, chosen_width, chosen_height});
    out << "<text x=\"" << format_svg(chosen_x) << "\" y=\"" << format_svg(chosen_y + 12.0) << "\">"
        << escape_text(primary) << "</text>\n";
    if (include_score) {
      out << "<text x=\"" << format_svg(chosen_x) << "\" y=\"" << format_svg(chosen_y + 12.0 + line_height)
          << "\">" << escape_text(score) << "</text>\n";
    }
  }

  void emit_hud(std::ostream& out, const Scene& scene) const {
    out << "<g id=\"hud\" font-family=\"monospace\" font-size=\"13\" fill=\"#E6EDF3\">\n";
    if (!config_.draw_hud) {
      out << "</g>\n";
      return;
    }
    const LabelBox box = hud_box(scene);
    out << "<rect x=\"" << format_svg(box.x) << "\" y=\"" << format_svg(box.y) << "\" width=\""
        << format_svg(box.width) << "\" height=\"" << format_svg(box.height)
        << "\" rx=\"6\" fill=\"#0B1220\" fill-opacity=\"0.82\" stroke=\"#263449\"/>\n";
    double y = box.y + 18.0;
    const double x = box.x + 10.0;
    out << "<text x=\"" << format_svg(x) << "\" y=\"" << format_svg(y) << "\">t = " << format_hud(scene.timestamp)
        << " s</text>\n";
    y += 16.0;
    out << "<text x=\"" << format_svg(x) << "\" y=\"" << format_svg(y) << "\">tracks " << scene.hud.tracks
        << "  C/T/L " << scene.hud.confirmed << "/" << scene.hud.tentative << "/" << scene.hud.lost
        << "</text>\n";
    if (scene.hud.has_metrics) {
      y += 16.0;
      out << "<text x=\"" << format_svg(x) << "\" y=\"" << format_svg(y) << "\">evaluated "
          << scene.hud.evaluated << "  mean minADE " << format_hud(scene.hud.mean_min_ade)
          << " m</text>\n";
      y += 16.0;
      out << "<text x=\"" << format_svg(x) << "\" y=\"" << format_svg(y) << "\">mean minFDE "
          << format_hud(scene.hud.mean_min_fde) << " m  miss " << format_hud(scene.hud.miss_percent)
          << "%</text>\n";
    }
    out << "</g>\n";
  }

  static std::string escape_text(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char character : text) {
      if (character == '&') {
        escaped += "&amp;";
      } else if (character == '<') {
        escaped += "&lt;";
      } else if (character == '>') {
        escaped += "&gt;";
      } else {
        escaped.push_back(character);
      }
    }
    return escaped;
  }

  VisualizerConfig config_{};
};

}  // namespace prediction
