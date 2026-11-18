#include "prediction/metrics.hpp"
#include "prediction/visualizer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

prediction::TrackState make_track(std::uint64_t id, double x, double y, double vx, double vy, double yaw,
                                  prediction::TrackStatus status = prediction::TrackStatus::Confirmed) {
  prediction::TrackState track;
  track.track_id = id;
  track.status = status;
  track.size = Eigen::Vector2d(4.0, 2.0);
  track.state << x, y, vx, vy, yaw, 0.0;
  return track;
}

prediction::TrajectoryPoint make_point(double timestamp, double x, double y, double vx = 0.0, double vy = 0.0) {
  prediction::TrajectoryPoint point;
  point.timestamp = timestamp;
  point.position = Eigen::Vector2d(x, y);
  point.velocity = Eigen::Vector2d(vx, vy);
  point.heading = 0.0;
  point.position_covariance = Eigen::Matrix2d::Identity() * 0.01;
  return point;
}

prediction::PredictedTrajectory make_modes(std::uint64_t id,
                                           const std::vector<std::vector<prediction::TrajectoryPoint>>& modes) {
  prediction::PredictedTrajectory prediction;
  prediction.track_id = id;
  prediction.modes.resize(modes.size());
  for (std::size_t index = 0; index < modes.size(); ++index) {
    prediction.modes[index].probability = 1.0 / static_cast<double>(modes.size());
    prediction.modes[index].waypoints = modes[index];
  }
  return prediction;
}

std::string render(const prediction::BevVisualizer& visualizer, const std::vector<prediction::TrackState>& tracks = {},
                   const std::vector<prediction::PredictedTrajectory>& predictions = {},
                   const std::vector<prediction::GroundTruthTrajectory>& truths = {},
                   const std::vector<prediction::TrajectoryMetrics>& metrics = {}, double timestamp = 1.25) {
  return visualizer.render_frame(timestamp, tracks, predictions, truths, metrics);
}

std::size_t layer_position(const std::string& svg, const std::string& id) {
  const std::string token = "id=\"" + id + "\"";
  const std::size_t position = svg.find(token);
  EXPECT_NE(position, std::string::npos) << id;
  return position;
}

std::string group_body(const std::string& svg, const std::string& id) {
  const std::string open = "<g id=\"" + id + "\"";
  const std::size_t start = svg.find(open);
  EXPECT_NE(start, std::string::npos) << id;
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t content = svg.find('>', start);
  std::size_t depth = 1U;
  std::size_t cursor = content + 1;
  while (depth != 0U && cursor < svg.size()) {
    const std::size_t next_open = svg.find("<g", cursor);
    const std::size_t next_close = svg.find("</g>", cursor);
    if (next_close == std::string::npos) {
      break;
    }
    if (next_open != std::string::npos && next_open < next_close) {
      ++depth;
      cursor = next_open + 2;
      continue;
    }
    --depth;
    cursor = next_close + 4;
  }
  return svg.substr(content + 1, cursor - 4 - (content + 1));
}

double attribute_number(const std::string& element, const std::string& name) {
  const std::string key = name + "=\"";
  const std::size_t start = element.find(key);
  EXPECT_NE(start, std::string::npos) << name << " in " << element;
  const std::size_t value = start + key.size();
  const std::size_t end = element.find('"', value);
  return std::stod(element.substr(value, end - value));
}

std::string element_after(const std::string& svg, const std::string& marker, const std::string& tag) {
  const std::size_t from = svg.find(marker);
  EXPECT_NE(from, std::string::npos) << marker;
  const std::size_t start = svg.find("<" + tag, from);
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = svg.find('>', start);
  return svg.substr(start, end - start + 1);
}

}  // namespace

namespace prediction {
namespace test_support {
const BevVisualizer& visualizer_linkage_anchor();
std::string visualizer_linkage_render();
}  // namespace test_support
}  // namespace prediction

TEST(BevVisualizer, ConfigValidationNamesTheOffendingField) {
  auto expect_field = [](prediction::VisualizerConfig config, const char* field) {
    try {
      prediction::BevVisualizer visualizer(config);
      FAIL() << "expected rejection of " << field;
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find(field), std::string::npos) << error.what();
    }
  };

  prediction::VisualizerConfig config;
  config.width_pixels = 0.0;
  expect_field(config, "VisualizerConfig::width_pixels");
  config = {};
  config.height_pixels = -1.0;
  expect_field(config, "VisualizerConfig::height_pixels");
  config = {};
  config.pixels_per_meter = std::numeric_limits<double>::quiet_NaN();
  expect_field(config, "VisualizerConfig::pixels_per_meter");
  config = {};
  config.center_world = Eigen::Vector2d(std::numeric_limits<double>::infinity(), 0.0);
  expect_field(config, "VisualizerConfig::center_world");
  config = {};
  config.grid_spacing_meters = -5.0;
  expect_field(config, "VisualizerConfig::grid_spacing_meters");
  config = {};
  config.covariance_sigma_scale = 0.0;
  expect_field(config, "VisualizerConfig::covariance_sigma_scale");
  config = {};
  config.ellipse_step_stride = 0U;
  expect_field(config, "VisualizerConfig::ellipse_step_stride");
  config = {};
  config.velocity_arrow_seconds = -0.1;
  expect_field(config, "VisualizerConfig::velocity_arrow_seconds");
  config = {};
  config.velocity_arrow_seconds = std::numeric_limits<double>::infinity();
  expect_field(config, "VisualizerConfig::velocity_arrow_seconds");

  config = {};
  config.draw_grid = false;
  config.grid_spacing_meters = std::numeric_limits<double>::quiet_NaN();
  expect_field(config, "VisualizerConfig::grid_spacing_meters");

  config = {};
  config.draw_covariance_ellipses = false;
  config.covariance_sigma_scale = -2.0;
  expect_field(config, "VisualizerConfig::covariance_sigma_scale");

  prediction::VisualizerConfig accepted;
  accepted.velocity_arrow_seconds = 0.0;
  accepted.width_pixels = 10.5;
  EXPECT_NO_THROW(prediction::BevVisualizer{accepted});
  EXPECT_DOUBLE_EQ(prediction::BevVisualizer{accepted}.config().velocity_arrow_seconds, 0.0);
}

TEST(BevVisualizer, EmptyInputsStillEmitAStandaloneSvg) {
  const prediction::BevVisualizer visualizer;
  const std::string svg = render(visualizer);
  EXPECT_NE(svg.find("<svg xmlns=\"http://www.w3.org/2000/svg\""), std::string::npos);
  EXPECT_NE(svg.find("width=\"1000\""), std::string::npos);
  EXPECT_NE(svg.find("viewBox=\"0 0 1000 1000\""), std::string::npos);
  EXPECT_NE(svg.find("clip-path=\"url(#viewport_clip)\""), std::string::npos);
  EXPECT_EQ(svg, render(visualizer));
  EXPECT_EQ(layer_position(svg, "background"), svg.find("id=\"background\""));
  EXPECT_LT(layer_position(svg, "background"), layer_position(svg, "grid"));
  EXPECT_LT(layer_position(svg, "grid"), layer_position(svg, "ground_truth"));
  EXPECT_LT(layer_position(svg, "ground_truth"), layer_position(svg, "uncertainty"));
  EXPECT_LT(layer_position(svg, "uncertainty"), layer_position(svg, "predictions"));
  EXPECT_LT(layer_position(svg, "predictions"), layer_position(svg, "tracks"));
  EXPECT_LT(layer_position(svg, "tracks"), layer_position(svg, "labels"));
  EXPECT_LT(layer_position(svg, "labels"), layer_position(svg, "hud"));
}

TEST(BevVisualizer, NonSquareOffCenterProjection) {
  prediction::VisualizerConfig config;
  config.width_pixels = 200.0;
  config.height_pixels = 80.0;
  config.pixels_per_meter = 2.0;
  config.center_world = Eigen::Vector2d(10.0, -4.0);
  config.draw_grid = false;
  config.draw_hud = false;
  config.draw_covariance_ellipses = false;
  const prediction::BevVisualizer visualizer(config);

  prediction::GroundTruthTrajectory truth;
  truth.track_id = 3;
  prediction::GroundTruthWaypoint waypoint;
  waypoint.timestamp = 0.0;
  waypoint.position = Eigen::Vector2d(11.0, -4.0);
  truth.waypoints = {waypoint};
  const std::string svg = render(visualizer, {}, {}, {truth});
  const std::string dot = element_after(svg, "id=\"ground_truth\"", "circle");
  ASSERT_FALSE(dot.empty());
  EXPECT_NEAR(attribute_number(dot, "cx"), 102.0, 1e-9);
  EXPECT_NEAR(attribute_number(dot, "cy"), 40.0, 1e-9);
}

TEST(BevVisualizer, NorthFacingBoxHasUpwardNoseAndEastwardVelocityArrow) {
  prediction::VisualizerConfig config;
  config.width_pixels = 200.0;
  config.height_pixels = 200.0;
  config.pixels_per_meter = 10.0;
  config.draw_grid = false;
  config.draw_hud = false;
  const prediction::BevVisualizer visualizer(config);
  const prediction::TrackState track =
      make_track(1, 0.0, 0.0, 10.0, 0.0, prediction::kPi / 2.0, prediction::TrackStatus::Confirmed);
  const std::string svg = render(visualizer, {track});
  const std::string group = element_after(svg, "id=\"tracks\"", "g");
  const std::size_t rotate_at = group.find("rotate(");
  ASSERT_NE(rotate_at, std::string::npos) << group;
  const double yaw = std::stod(group.substr(rotate_at + 7));
  EXPECT_NEAR(yaw, -90.0, 1e-9) << group;
  const std::string nose = element_after(svg, "rotate(", "line");
  EXPECT_NEAR(attribute_number(nose, "x2"), 20.0, 1e-9) << nose;

  const std::string tracks = group_body(svg, "tracks");
  const std::size_t group_end = tracks.rfind("</g>");
  const std::size_t arrow = tracks.find("<line", group_end);
  ASSERT_NE(arrow, std::string::npos);
  const std::string arrow_element = tracks.substr(arrow, tracks.find('>', arrow) - arrow + 1);
  EXPECT_NEAR(attribute_number(arrow_element, "x1"), 100.0, 1e-6);
  EXPECT_NEAR(attribute_number(arrow_element, "y1"), 100.0, 1e-6);
  EXPECT_GT(attribute_number(arrow_element, "x2"), attribute_number(arrow_element, "x1"));
  EXPECT_NEAR(attribute_number(arrow_element, "y2"), 100.0, 1e-6);
}

TEST(BevVisualizer, AnisotropicCovarianceHasNegativeScreenSlope) {
  prediction::VisualizerConfig config;
  config.ellipse_step_stride = 1U;
  config.draw_grid = false;
  config.draw_hud = false;
  const prediction::BevVisualizer visualizer(config);
  prediction::TrajectoryPoint point = make_point(0.1, 0.0, 0.0);
  point.position_covariance << 2.0, 1.0, 1.0, 2.0;
  const prediction::PredictedTrajectory prediction = make_modes(4, {{point}});
  const std::string svg = render(visualizer, {}, {prediction});
  const std::string ellipse = element_after(svg, "id=\"uncertainty\"", "ellipse");
  ASSERT_FALSE(ellipse.empty());
  const std::size_t rotate = ellipse.find("rotate(");
  ASSERT_NE(rotate, std::string::npos);
  const double angle = std::stod(ellipse.substr(rotate + 7));
  const double folded = std::sin((angle - (-45.0)) * prediction::kPi / 180.0);
  EXPECT_NEAR(folded, 0.0, 1e-8) << ellipse;
  EXPECT_LT(angle, 0.0) << ellipse;
  EXPECT_GT(attribute_number(ellipse, "rx"), attribute_number(ellipse, "ry"));
}

TEST(BevVisualizer, CovarianceRankAndRejectionCases) {
  prediction::VisualizerConfig config;
  config.ellipse_step_stride = 1U;
  config.draw_grid = false;
  const prediction::BevVisualizer visualizer(config);

  prediction::TrajectoryPoint isotropic = make_point(0.1, 0.0, 0.0);
  isotropic.position_covariance = Eigen::Matrix2d::Identity() * 0.25;
  const std::string iso = render(visualizer, {}, {make_modes(1, {{isotropic}})});
  const std::string iso_ellipse = element_after(iso, "id=\"uncertainty\"", "ellipse");
  EXPECT_NEAR(std::stod(iso_ellipse.substr(iso_ellipse.find("rotate(") + 7)), 0.0, 1e-9);
  EXPECT_NEAR(attribute_number(iso_ellipse, "rx"), attribute_number(iso_ellipse, "ry"), 1e-8);

  prediction::TrajectoryPoint rank_one = make_point(0.1, 1.0, 0.0);
  rank_one.position_covariance << 1.0, 0.0, 0.0, 0.0;
  const std::string line_svg = render(visualizer, {}, {make_modes(1, {{rank_one}})});
  EXPECT_EQ(element_after(line_svg, "id=\"uncertainty\"", "ellipse"), "");
  const std::string segment = element_after(line_svg, "id=\"uncertainty\"", "line");
  ASSERT_FALSE(segment.empty());
  EXPECT_NE(attribute_number(segment, "x1"), attribute_number(segment, "x2"));

  prediction::TrajectoryPoint zero = make_point(0.1, 2.0, 0.0);
  zero.position_covariance = Eigen::Matrix2d::Zero();
  const std::string zero_svg = render(visualizer, {}, {make_modes(1, {{zero}})});
  EXPECT_EQ(group_body(zero_svg, "uncertainty").find("<ellipse"), std::string::npos);
  EXPECT_EQ(group_body(zero_svg, "uncertainty").find("<line"), std::string::npos);

  prediction::TrajectoryPoint barely = make_point(0.1, 0.0, 0.0);
  barely.position_covariance << 1.0, 0.0, 0.0, -1e-15;
  EXPECT_NO_THROW(render(visualizer, {}, {make_modes(1, {{barely}})}));

  prediction::TrajectoryPoint indefinite = make_point(0.1, 0.0, 0.0);
  indefinite.position_covariance << 1.0, 0.0, 0.0, -1.0;
  EXPECT_THROW(render(visualizer, {}, {make_modes(1, {{indefinite}})}), std::invalid_argument);

  prediction::TrajectoryPoint asymmetric = make_point(0.1, 0.0, 0.0);
  asymmetric.position_covariance << 1.0, 0.2, -0.2, 1.0;
  EXPECT_THROW(render(visualizer, {}, {make_modes(1, {{asymmetric}})}), std::invalid_argument);

  prediction::TrajectoryPoint non_finite = make_point(0.1, 0.0, 0.0);
  non_finite.position_covariance << 1.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(render(visualizer, {}, {make_modes(1, {{non_finite}})}), std::invalid_argument);
}

TEST(BevVisualizer, InvalidCovarianceIsCheckedWhenNotDrawn) {
  prediction::VisualizerConfig hidden;
  hidden.draw_covariance_ellipses = false;
  hidden.ellipse_step_stride = 2U;
  const prediction::BevVisualizer visualizer(hidden);
  prediction::TrajectoryPoint skipped = make_point(0.1, 0.0, 0.0);
  prediction::TrajectoryPoint later = make_point(0.2, 1.0, 0.0);
  skipped.position_covariance << 1.0, 0.0, 0.0, -1.0;
  EXPECT_THROW(render(visualizer, {}, {make_modes(9, {{later, skipped}})}), std::invalid_argument);

  prediction::VisualizerConfig drawn;
  drawn.ellipse_step_stride = 2U;
  const prediction::BevVisualizer stride_visualizer(drawn);
  EXPECT_THROW(render(stride_visualizer, {}, {make_modes(9, {{skipped, later}})}), std::invalid_argument);
  prediction::TrajectoryPoint skipped_sample = make_point(0.3, 3.0, 0.0);
  const std::string stride_body = group_body(
      render(stride_visualizer, {}, {make_modes(9, {{later, skipped_sample}})}), "uncertainty");
  EXPECT_NE(stride_body.find("cx=\"520\""), std::string::npos) << stride_body;
  EXPECT_EQ(stride_body.find("cx=\"560\""), std::string::npos) << stride_body;
}

TEST(BevVisualizer, ModesAreAuthoritativeAndLegacyIsAFallback) {
  prediction::VisualizerConfig config;
  config.draw_covariance_ellipses = false;
  config.draw_grid = false;
  config.draw_hud = false;
  config.pixels_per_meter = 1.0;
  const prediction::BevVisualizer visualizer(config);

  prediction::PredictedTrajectory prediction = make_modes(1, {{make_point(0.1, 0.0, 0.0), make_point(0.2, 4.0, 0.0)}});
  prediction::PredictedWaypoint corrupt;
  corrupt.timestamp = std::numeric_limits<double>::quiet_NaN();
  corrupt.position = Eigen::Vector2d(80.0, 80.0);
  prediction.waypoints = {corrupt};
  const std::string svg = render(visualizer, {}, {prediction});
  EXPECT_EQ(svg.find("80,"), std::string::npos);
  EXPECT_EQ(svg.find(",80"), std::string::npos);
  EXPECT_NE(group_body(svg, "predictions").find("#60A5FA"), std::string::npos);

  prediction::PredictedTrajectory legacy;
  legacy.track_id = 2;
  prediction::PredictedWaypoint only;
  only.timestamp = 0.2;
  only.position = Eigen::Vector2d(3.0, -1.0);
  only.heading = 0.0;
  legacy.waypoints = {only};
  const std::string legacy_svg = render(visualizer, {}, {legacy});
  EXPECT_NE(group_body(legacy_svg, "predictions").find("data-mode=\"legacy\""), std::string::npos);
}

TEST(BevVisualizer, ExtraModesAndSinglePointPaths) {
  prediction::VisualizerConfig config;
  config.draw_covariance_ellipses = false;
  config.draw_grid = false;
  const prediction::BevVisualizer visualizer(config);
  prediction::PredictedTrajectory prediction =
      make_modes(1, {{make_point(0.1, 0.0, 0.0), make_point(0.2, 1.0, 0.0)},
                     {make_point(0.1, 0.0, 1.0), make_point(0.2, 1.0, 1.0)},
                     {make_point(0.1, 0.0, 2.0), make_point(0.2, 1.0, 2.0)},
                     {make_point(0.1, 0.0, 3.0)}});
  const std::string svg = render(visualizer, {}, {prediction});
  const std::string predictions = group_body(svg, "predictions");
  EXPECT_NE(predictions.find("#60A5FA"), std::string::npos);
  EXPECT_NE(predictions.find("#C084FC"), std::string::npos);
  EXPECT_NE(predictions.find("#22D3EE"), std::string::npos);
  EXPECT_NE(predictions.find("data-mode=\"3\""), std::string::npos);
  EXPECT_NE(predictions.find("#818CF8"), std::string::npos);
  EXPECT_NE(predictions.find("<circle"), std::string::npos);
}

TEST(BevVisualizer, WorldAnchoredGridSurvivesACameraShift) {
  prediction::VisualizerConfig left;
  left.width_pixels = 100.0;
  left.height_pixels = 100.0;
  left.pixels_per_meter = 2.0;
  left.grid_spacing_meters = 10.0;
  left.center_world = Eigen::Vector2d(0.0, 0.0);
  left.draw_hud = false;
  prediction::VisualizerConfig shifted = left;
  shifted.center_world = Eigen::Vector2d(3.0, 0.0);
  const std::string first = render(prediction::BevVisualizer(left));
  const std::string second = render(prediction::BevVisualizer(shifted));
  const std::string first_line = element_after(first, "id=\"grid\"", "line");
  const std::string second_line = element_after(second, "id=\"grid\"", "line");
  ASSERT_FALSE(first_line.empty());
  ASSERT_FALSE(second_line.empty());
  const double world_from_first =
      left.center_world.x() + (attribute_number(first_line, "x1") - left.width_pixels * 0.5) / left.pixels_per_meter;
  const double world_from_second = shifted.center_world.x() +
                                   (attribute_number(second_line, "x1") - shifted.width_pixels * 0.5) /
                                       shifted.pixels_per_meter;
  EXPECT_NEAR(std::fmod(world_from_first, left.grid_spacing_meters), 0.0, 1e-9);
  EXPECT_NEAR(std::fmod(world_from_second, shifted.grid_spacing_meters), 0.0, 1e-9);
  EXPECT_NE(attribute_number(first_line, "x1"), attribute_number(second_line, "x1"));
}

TEST(BevVisualizer, ExcessiveGridDensityOmitsTheGridAndExtremeCoordinatesFinish) {
  prediction::VisualizerConfig dense;
  dense.width_pixels = 1000.0;
  dense.height_pixels = 1000.0;
  dense.pixels_per_meter = 20.0;
  dense.grid_spacing_meters = 1e-6;
  const prediction::BevVisualizer visualizer(dense);
  const std::string svg = render(visualizer);
  EXPECT_NE(group_body(svg, "grid").find("visible line count exceeds 2048"), std::string::npos);
  EXPECT_EQ(group_body(svg, "grid").find("<line"), std::string::npos);

  prediction::VisualizerConfig extreme_camera = dense;
  extreme_camera.grid_spacing_meters = 5.0;
  extreme_camera.center_world = Eigen::Vector2d(1e300, -1e300);
  EXPECT_NO_THROW(render(prediction::BevVisualizer(extreme_camera)));

  prediction::TrackState far = make_track(1, 1e308, 0.0, 0.0, 0.0, 0.0);
  EXPECT_THROW(render(visualizer, {far}), std::overflow_error);
}

TEST(BevVisualizer, LayerTogglesAndDeterministicLocaleIndependentOutput) {
  prediction::VisualizerConfig config;
  config.draw_grid = false;
  config.draw_covariance_ellipses = false;
  config.draw_hud = false;
  config.pixels_per_meter = 1.0;
  const prediction::BevVisualizer visualizer(config);
  prediction::TrajectoryPoint point = make_point(0.1, 1.25, -0.5);
  prediction::GroundTruthTrajectory truth;
  truth.track_id = 7;
  prediction::GroundTruthWaypoint waypoint;
  waypoint.timestamp = 0.4;
  waypoint.position = Eigen::Vector2d(1.25, -0.5);
  truth.waypoints = {waypoint};
  const std::string svg =
      render(visualizer, {make_track(7, 0.0, 0.0, 1.0, 0.0, 0.0)}, {make_modes(7, {{point}})}, {truth});
  EXPECT_EQ(group_body(svg, "grid").find("<line"), std::string::npos);
  EXPECT_EQ(group_body(svg, "uncertainty").find("<ellipse"), std::string::npos);
  EXPECT_EQ(group_body(svg, "hud").find("<text"), std::string::npos);
  EXPECT_NE(group_body(svg, "tracks").find("<rect"), std::string::npos);
  EXPECT_EQ(svg, render(visualizer, {make_track(7, 0.0, 0.0, 1.0, 0.0, 0.0)}, {make_modes(7, {{point}})}, {truth}));

  struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
  };
  const std::locale injected(std::locale::classic(), new CommaDecimal);
  const std::locale previous = std::locale::global(injected);
  const std::string localized =
      render(visualizer, {make_track(7, 0.0, 0.0, 1.0, 0.0, 0.0)}, {make_modes(7, {{point}})}, {truth});
  std::locale::global(previous);
  EXPECT_EQ(localized, svg);
  EXPECT_EQ(localized.find("1,25"), std::string::npos);
  EXPECT_NE(localized.find("1.25"), std::string::npos);
}

TEST(BevVisualizer, HudAggregatesEveryMetricAndSuppressesAmbiguousScores) {
  prediction::VisualizerConfig config;
  config.width_pixels = 800.0;
  config.height_pixels = 800.0;
  config.draw_grid = false;
  config.draw_covariance_ellipses = false;
  const prediction::BevVisualizer visualizer(config);
  std::vector<prediction::TrackState> tracks = {
      make_track(1, -10.0, -10.0, 0.0, 0.0, 0.0, prediction::TrackStatus::Confirmed),
      make_track(2, 10.0, -10.0, 0.0, 0.0, 0.0, prediction::TrackStatus::Tentative),
      make_track(2, 10.0, 10.0, 0.0, 0.0, 0.0, prediction::TrackStatus::Lost),
  };
  prediction::TrajectoryMetrics first;
  first.track_id = 1;
  first.min_ade = 1.0;
  first.min_fde = 2.0;
  first.is_miss = false;
  prediction::TrajectoryMetrics second = first;
  second.track_id = 9;
  second.min_ade = 3.0;
  second.min_fde = 5.0;
  second.is_miss = true;
  prediction::TrajectoryMetrics duplicate = first;
  duplicate.track_id = 2;
  duplicate.min_ade = 4.0;
  duplicate.min_fde = 4.0;
  duplicate.is_miss = true;
  const std::string svg = render(visualizer, tracks, {}, {}, {first, second, duplicate, duplicate});
  const std::string hud = group_body(svg, "hud");
  EXPECT_NE(hud.find("tracks 3"), std::string::npos);
  EXPECT_NE(hud.find("C/T/L 1/1/1"), std::string::npos);
  EXPECT_NE(hud.find("evaluated 4"), std::string::npos);
  EXPECT_NE(hud.find("mean minADE 3.000"), std::string::npos);
  EXPECT_NE(hud.find("mean minFDE 3.750"), std::string::npos);
  EXPECT_NE(hud.find("miss 75.000%"), std::string::npos);
  EXPECT_NE(svg.find("#1 Confirmed"), std::string::npos);
  EXPECT_NE(svg.find("1.000 / 2.000"), std::string::npos);
  EXPECT_EQ(svg.find("4.000 / 4.000"), std::string::npos);

  const std::string unevaluated = render(visualizer, {tracks[0]});
  EXPECT_EQ(group_body(unevaluated, "hud").find("mean minADE"), std::string::npos);
  EXPECT_NE(group_body(unevaluated, "hud").find("tracks 1"), std::string::npos);
}

TEST(BevVisualizer, RejectsMalformedTracksAndMetricsWithContext) {
  const prediction::BevVisualizer visualizer;
  prediction::TrackState bad = make_track(4, 0.0, 0.0, 0.0, 0.0, 0.0);
  bad.size.y() = 0.0;
  try {
    render(visualizer, {bad});
    FAIL() << "expected invalid size";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("track_id 4"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("size.y"), std::string::npos);
  }

  bad = make_track(4, 0.0, 0.0, 0.0, 0.0, 0.0);
  bad.status = static_cast<prediction::TrackStatus>(9);
  EXPECT_THROW(render(visualizer, {bad}), std::invalid_argument);
  bad = make_track(4, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0, 0.0);
  EXPECT_THROW(render(visualizer, {bad}), std::invalid_argument);

  prediction::TrajectoryMetrics metrics;
  metrics.track_id = 4;
  metrics.min_ade = -1.0;
  EXPECT_THROW(render(visualizer, {}, {}, {}, {metrics}), std::invalid_argument);
  EXPECT_THROW(render(visualizer, {}, {}, {}, {}, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(BevVisualizer, StatusStylesAndInputOrderAreStable) {
  prediction::VisualizerConfig config;
  config.draw_grid = false;
  config.draw_hud = false;
  config.velocity_arrow_seconds = 0.0;
  const prediction::BevVisualizer visualizer(config);
  const std::vector<prediction::TrackState> tracks = {
      make_track(8, -5.0, 0.0, 0.0, 0.0, 0.0, prediction::TrackStatus::Lost),
      make_track(3, 0.0, 0.0, 0.0, 0.0, 0.0, prediction::TrackStatus::Tentative),
      make_track(11, 5.0, 0.0, 0.0, 0.0, 0.0, prediction::TrackStatus::Confirmed),
  };
  const std::string svg = render(visualizer, tracks);
  const std::string body = svg;
  const std::size_t lost = body.find("#FB7185");
  const std::size_t tentative = body.find("#FBBF24");
  const std::size_t confirmed = body.find("#34D399");
  EXPECT_LT(lost, tentative);
  EXPECT_LT(tentative, confirmed);
  EXPECT_NE(body.find("stroke-dasharray=\"10 5\""), std::string::npos);
  EXPECT_NE(body.find("stroke-dasharray=\"4 3\""), std::string::npos);
}

TEST(BevVisualizer, TrajectoryMetricsIsVisibleThroughMetricsAndTypes) {
  prediction::TrajectoryMetrics from_metrics;
  from_metrics.track_id = 6;
  prediction::TrajectoryMetrics from_types = from_metrics;
  EXPECT_EQ(from_types.track_id, 6U);
  EXPECT_FALSE(from_types.is_miss);
}

TEST(BevVisualizer, HeaderOnlyAnchorLinksFromAnotherTranslationUnit) {
  const prediction::BevVisualizer& anchor = prediction::test_support::visualizer_linkage_anchor();
  EXPECT_DOUBLE_EQ(anchor.config().width_pixels, 1000.0);
  EXPECT_EQ(anchor.config().ellipse_step_stride, 5U);
  const std::string other = prediction::test_support::visualizer_linkage_render();
  EXPECT_EQ(other, render(prediction::BevVisualizer{}, {}, {}, {}, {}, 0.0));
}
