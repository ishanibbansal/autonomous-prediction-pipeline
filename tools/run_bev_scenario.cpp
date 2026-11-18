// Deterministic intersection scenario for the bird's-eye-view renderer.
//
// Three analytic actors share one identity ego pose. The tracker, predictor,
// and evaluator are the real library types. Ground truth is sampled from the
// actor motion, never copied from a forecast. The third actor drops out for
// two frames after confirmation, fewer than max_consecutive_misses, so the
// tracker coasts it as Lost and then recovers the same track_id.

#include "prediction/metrics.hpp"
#include "prediction/predictor.hpp"
#include "prediction/tracker.hpp"
#include "prediction/visualizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kDt = 0.1;
constexpr std::uint64_t kFrameCount = 40;
constexpr std::size_t kDropoutActor = 2U;
constexpr double kTurnStart = 1.0;
constexpr double kTurnRate = 0.05;
constexpr double kBoxLength = 4.5;
constexpr double kBoxWidth = 1.8;

struct PoseSample {
  double x{0.0};
  double y{0.0};
  double vx{0.0};
  double vy{0.0};
  double heading{0.0};
};

struct Actor {
  const char* name;
  PoseSample (*sample)(double time);
};

PoseSample sample_cruiser(double time) {
  PoseSample pose;
  pose.x = -30.0 + 8.0 * time;
  pose.y = -8.0;
  pose.vx = 8.0;
  pose.vy = 0.0;
  pose.heading = 0.0;
  return pose;
}

PoseSample sample_turner(double time) {
  const double speed = 6.0;
  PoseSample pose;
  if (time <= kTurnStart) {
    pose.x = -2.0;
    pose.y = -28.0 + speed * time;
    pose.vx = 0.0;
    pose.vy = speed;
    pose.heading = prediction::kPi / 2.0;
    return pose;
  }
  const double elapsed = time - kTurnStart;
  const double theta = kTurnRate * elapsed;
  const double center_x = -2.0 - speed / kTurnRate;
  const double center_y = -28.0 + speed * kTurnStart;
  const double start_heading = prediction::kPi / 2.0;
  pose.heading = prediction::normalize_angle(start_heading + theta);
  pose.vx = speed * std::cos(pose.heading);
  pose.vy = speed * std::sin(pose.heading);
  pose.x = center_x + (speed / kTurnRate) * std::sin(pose.heading);
  pose.y = center_y - (speed / kTurnRate) * std::cos(pose.heading);
  return pose;
}

PoseSample sample_dropout(double time) {
  PoseSample pose;
  pose.x = 24.0 - 7.0 * time;
  pose.y = 10.0;
  pose.vx = -7.0;
  pose.vy = 0.0;
  pose.heading = prediction::kPi;
  return pose;
}

const Actor kActors[] = {
    {"cruiser", sample_cruiser},
    {"left_turner", sample_turner},
    {"dropout", sample_dropout},
};

constexpr std::size_t kActorCount = sizeof(kActors) / sizeof(kActors[0]);

bool actor_is_visible(std::size_t actor, std::uint64_t frame) {
  if (actor != kDropoutActor) {
    return true;
  }
  return frame < 8U || frame >= 10U;
}

prediction::Detection detection_at(const PoseSample& pose) {
  prediction::Detection detection;
  detection.position = Eigen::Vector2d(pose.x, pose.y);
  detection.size = Eigen::Vector2d(kBoxLength, kBoxWidth);
  detection.heading = pose.heading;
  detection.score = 0.9;
  return detection;
}

prediction::DetectionFrame make_frame(std::uint64_t frame) {
  prediction::DetectionFrame packet;
  packet.frame_id = frame;
  packet.timestamp = static_cast<double>(frame) * kDt;
  packet.ego_pose_2d = prediction::Pose2D{0.0, 0.0, 0.0};
  packet.ego_pose_se3 = Eigen::Matrix4d::Identity();
  for (std::size_t actor = 0; actor < kActorCount; ++actor) {
    if (!actor_is_visible(actor, frame)) {
      continue;
    }
    packet.detections.push_back(detection_at(kActors[actor].sample(packet.timestamp)));
  }
  return packet;
}

int fail(const std::string& message) {
  std::cerr << "bev_demo: " << message << "\n";
  return 1;
}

std::filesystem::path frame_path(const std::filesystem::path& directory, std::uint64_t frame) {
  std::string name = "frame_";
  const std::string digits = std::to_string(frame);
  name.append(4U - digits.size(), '0');
  name += digits;
  name += ".svg";
  return directory / name;
}

bool directory_has_stale_frames(const std::filesystem::path& directory) {
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.rfind("frame_", 0) == 0 && entry.path().extension() == ".svg") {
      return true;
    }
  }
  return false;
}

std::size_t nearest_actor(const prediction::TrackState& track, double time) {
  std::size_t best = 0U;
  double best_distance = std::numeric_limits<double>::infinity();
  for (std::size_t actor = 0; actor < kActorCount; ++actor) {
    const PoseSample pose = kActors[actor].sample(time);
    const double distance = std::hypot(track.state(0) - pose.x, track.state(1) - pose.y);
    if (distance < best_distance) {
      best_distance = distance;
      best = actor;
    }
  }
  return best;
}

prediction::GroundTruthTrajectory truth_for(std::uint64_t track_id, std::size_t actor,
                                            const prediction::PredictedTrajectory& prediction) {
  prediction::GroundTruthTrajectory truth;
  truth.track_id = track_id;
  if (prediction.modes.empty()) {
    throw std::runtime_error("predictor returned no modes for track " + std::to_string(track_id));
  }
  const std::vector<prediction::TrajectoryPoint>& samples = prediction.modes.front().waypoints;
  truth.waypoints.reserve(samples.size());
  for (const prediction::TrajectoryPoint& sample : samples) {
    const PoseSample pose = kActors[actor].sample(sample.timestamp);
    prediction::GroundTruthWaypoint waypoint;
    waypoint.timestamp = sample.timestamp;
    waypoint.position = Eigen::Vector2d(pose.x, pose.y);
    truth.waypoints.push_back(waypoint);
  }
  return truth;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return fail("usage: bev_demo <output-directory>");
  }

  const std::filesystem::path output(argv[1]);
  std::error_code error;
  std::filesystem::create_directories(output, error);
  if (error) {
    return fail("cannot create output directory: " + error.message());
  }
  if (!std::filesystem::is_directory(output)) {
    return fail("output path is not a directory");
  }
  try {
    if (directory_has_stale_frames(output)) {
      return fail("output directory already contains frame_*.svg; refusing to mix stale frames");
    }
  } catch (const std::filesystem::filesystem_error& exception) {
    return fail(std::string("cannot inspect output directory: ") + exception.what());
  }

  prediction::TrackerConfig tracker_config;
  tracker_config.output_policy = prediction::TrackOutputPolicy::All;
  tracker_config.filter_config.sigma_a_sq = 4.0;
  prediction::MultiObjectTracker tracker(tracker_config);
  prediction::KinematicTrajectoryPredictor predictor;
  prediction::TrajectoryEvaluator evaluator;
  prediction::VisualizerConfig view;
  view.pixels_per_meter = 8.0;
  view.center_world = Eigen::Vector2d::Zero();
  view.grid_spacing_meters = 10.0;
  const prediction::BevVisualizer visualizer(view);

  std::vector<std::uint64_t> actor_track(kActorCount, 0U);
  std::vector<prediction::TrackStatus> dropout_history;
  bool saw_evaluated_hud = false;

  try {
    for (std::uint64_t frame = 0; frame < kFrameCount; ++frame) {
      const prediction::DetectionFrame packet = make_frame(frame);
      const std::vector<prediction::TrackState>& tracks = tracker.process_frame(packet);
      const double time = packet.timestamp;

      std::vector<std::size_t> matched(kActorCount, 0U);
      for (const prediction::TrackState& track : tracks) {
        const std::size_t actor = nearest_actor(track, time);
        const PoseSample pose = kActors[actor].sample(time);
        const double distance = std::hypot(track.state(0) - pose.x, track.state(1) - pose.y);
        if (!(distance < 8.0)) {
          return fail("frame " + std::to_string(frame) + " track " + std::to_string(track.track_id) +
                      " is not near any actor");
        }
        ++matched[actor];
        if (actor_track[actor] == 0U) {
          actor_track[actor] = track.track_id;
        } else if (actor_track[actor] != track.track_id) {
          return fail("actor " + std::string(kActors[actor].name) + " changed track_id from " +
                      std::to_string(actor_track[actor]) + " to " + std::to_string(track.track_id));
        }
        if (actor == kDropoutActor) {
          dropout_history.push_back(track.status);
        }
      }
      for (std::size_t actor = 0; actor < kActorCount; ++actor) {
        if (actor_is_visible(actor, frame) && matched[actor] != 1U) {
          return fail("frame " + std::to_string(frame) + " actor " + kActors[actor].name +
                      " was not uniquely tracked");
        }
      }

      const std::vector<prediction::PredictedTrajectory> predictions = predictor.predict(tracks, time);
      std::vector<prediction::GroundTruthTrajectory> truths;
      truths.reserve(predictions.size());
      for (const prediction::PredictedTrajectory& prediction : predictions) {
        const auto actor = std::find(actor_track.begin(), actor_track.end(), prediction.track_id);
        if (actor == actor_track.end()) {
          return fail("prediction track_id has no actor association");
        }
        truths.push_back(
            truth_for(prediction.track_id, static_cast<std::size_t>(actor - actor_track.begin()), prediction));
      }

      std::vector<prediction::TrajectoryMetrics> metrics;
      if (!predictions.empty()) {
        const prediction::BatchMetricsSummary summary = evaluator.evaluate_batch(predictions, truths);
        metrics = summary.per_track;
      }

      const std::string svg = visualizer.render_frame(time, tracks, predictions, truths, metrics);
      if (svg.find("<svg") == std::string::npos || svg.find("evaluated") == std::string::npos) {
        return fail("frame " + std::to_string(frame) + " did not produce an evaluated SVG");
      }
      saw_evaluated_hud = true;

      const std::filesystem::path path = frame_path(output, frame);
      std::ofstream file(path);
      if (!file) {
        return fail("cannot open " + path.string());
      }
      file << svg;
      if (!file) {
        return fail("failed while writing " + path.string());
      }
    }
  } catch (const std::exception& exception) {
    return fail(exception.what());
  }

  if (actor_track[kDropoutActor] == 0U) {
    return fail("dropout actor never received a track_id");
  }
  const std::vector<prediction::TrackStatus> expected = {
      prediction::TrackStatus::Tentative, prediction::TrackStatus::Tentative,
      prediction::TrackStatus::Confirmed,  prediction::TrackStatus::Lost,
      prediction::TrackStatus::Confirmed};
  std::size_t cursor = 0U;
  for (const prediction::TrackStatus status : dropout_history) {
    if (cursor < expected.size() && status == expected[cursor]) {
      ++cursor;
    }
  }
  if (cursor != expected.size()) {
    return fail("dropout actor did not show Tentative -> Confirmed -> Lost -> Confirmed on one track_id");
  }
  if (!saw_evaluated_hud) {
    return fail("no frame produced an evaluated HUD");
  }

  std::cout << "bev_demo: wrote " << kFrameCount << " frames to " << output.string() << " (dropout track_id "
            << actor_track[kDropoutActor] << ")\n";
  return 0;
}
