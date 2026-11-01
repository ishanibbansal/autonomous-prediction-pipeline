#pragma once

#include "prediction/types.hpp"

#include <cmath>
#include <istream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prediction {

/// Streaming reader for the unquoted, eleven-column perception CSV schema.
/// The stream must outlive the reader. Rows are assumed ordered by frame_id.
class CsvReader {
 public:
  explicit CsvReader(std::istream& stream) : stream_(stream) {}

  /// Returns one consecutive frame group; leaves out_frame unchanged on failure.
  bool read_next_frame(DetectionFrame& out_frame) {
    if (exhausted_) {
      return false;
    }
    if (!header_checked_) {
      header_checked_ = true;
      std::string line;
      if (!next_nonblank_line(line) || split(line) != header()) {
        exhausted_ = true;
        return false;
      }
    }

    std::optional<DetectionFrame> frame = std::move(pending_);
    pending_.reset();
    std::string line;
    while (next_nonblank_line(line)) {
      auto row = parse_row(line);
      if (!row) {
        continue;
      }
      if (!frame) {
        frame = std::move(row);
      } else if (row->frame_id != frame->frame_id) {
        pending_ = std::move(row);
        out_frame = std::move(*frame);
        return true;
      } else if (same_metadata(*frame, *row) && !row->detections.empty()) {
        frame->detections.push_back(std::move(row->detections.front()));
      }
    }
    exhausted_ = true;
    if (!frame) {
      return false;
    }
    out_frame = std::move(*frame);
    return true;
  }

  static std::vector<DetectionFrame> load_all_frames(std::istream& stream) {
    CsvReader reader(stream);
    std::vector<DetectionFrame> frames;
    DetectionFrame frame;
    while (reader.read_next_frame(frame)) {
      frames.push_back(std::move(frame));
    }
    return frames;
  }

 private:
  static std::string trim(const std::string& token) {
    const auto first = token.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos) {
      return {};
    }
    const auto last = token.find_last_not_of(" \t\r\n\f\v");
    return token.substr(first, last - first + 1);
  }

  static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::string::size_type start = 0;
    while (true) {
      const auto comma = line.find(',', start);
      fields.push_back(trim(line.substr(start, comma == std::string::npos
                                                  ? std::string::npos : comma - start)));
      if (comma == std::string::npos) {
        return fields;
      }
      start = comma + 1;
    }
  }

  static const std::vector<std::string>& header() {
    static const std::vector<std::string> names = {
        "frame_id", "timestamp", "ego_x", "ego_y", "ego_yaw", "det_x",
        "det_y", "size_x", "size_y", "heading", "score"};
    return names;
  }

  bool next_nonblank_line(std::string& line) {
    while (std::getline(stream_, line)) {
      if (!trim(line).empty()) {
        return true;
      }
    }
    return false;
  }

  static bool parse_finite(const std::string& token, double& value) {
    try {
      std::size_t consumed = 0;
      value = std::stod(token, &consumed);
      return consumed == token.size() && std::isfinite(value);
    } catch (const std::invalid_argument&) {
      return false;
    } catch (const std::out_of_range&) {
      return false;
    }
  }

  static bool parse_id(const std::string& token, std::uint64_t& value) {
    if (token.empty()) {
      return false;
    }
    // Accumulate directly in uint64_t so the range is independent of unsigned long long.
    std::uint64_t result = 0;
    constexpr std::uint64_t maximum = UINT64_MAX;
    for (const char digit : token) {
      if (digit < '0' || digit > '9') {
        return false;
      }
      const auto number = static_cast<std::uint64_t>(digit - '0');
      if (result > (maximum - number) / 10) {
        return false;
      }
      result = result * 10 + number;
    }
    value = result;
    return true;
  }

  static bool is_absent(const std::string& token) {
    return token.empty() ||
           (token.size() == 3 && (token[0] == 'n' || token[0] == 'N') &&
            (token[1] == 'a' || token[1] == 'A') && (token[2] == 'n' || token[2] == 'N'));
  }

  static bool same_metadata(const DetectionFrame& first, const DetectionFrame& row) {
    return first.timestamp == row.timestamp && first.ego_pose_2d.x == row.ego_pose_2d.x &&
           first.ego_pose_2d.y == row.ego_pose_2d.y &&
           first.ego_pose_2d.yaw == row.ego_pose_2d.yaw;
  }

  static std::optional<DetectionFrame> parse_row(const std::string& line) {
    const auto fields = split(line);
    DetectionFrame frame;
    if (fields.size() != 11 || !parse_id(fields[0], frame.frame_id) ||
        !parse_finite(fields[1], frame.timestamp) ||
        !parse_finite(fields[2], frame.ego_pose_2d.x) ||
        !parse_finite(fields[3], frame.ego_pose_2d.y) ||
        !parse_finite(fields[4], frame.ego_pose_2d.yaw)) {
      return std::nullopt;
    }

    bool sentinel = true;
    for (std::size_t index = 5; index < fields.size(); ++index) {
      sentinel = sentinel && is_absent(fields[index]);
    }
    if (!sentinel) {
      Detection detection;
      if (!parse_finite(fields[5], detection.position.x()) ||
          !parse_finite(fields[6], detection.position.y()) ||
          !parse_finite(fields[7], detection.size.x()) ||
          !parse_finite(fields[8], detection.size.y()) ||
          !parse_finite(fields[9], detection.heading) ||
          !parse_finite(fields[10], detection.score) ||
          detection.size.x() <= 0.0 || detection.size.y() <= 0.0) {
        return std::nullopt;
      }
      detection.heading = normalize_angle(detection.heading);
      frame.detections.push_back(std::move(detection));
    }

    const auto& pose = frame.ego_pose_2d;
    const double cosine = std::cos(pose.yaw);
    const double sine = std::sin(pose.yaw);
    frame.ego_pose_se3(0, 0) = cosine;
    frame.ego_pose_se3(0, 1) = -sine;
    frame.ego_pose_se3(1, 0) = sine;
    frame.ego_pose_se3(1, 1) = cosine;
    frame.ego_pose_se3(0, 3) = pose.x;
    frame.ego_pose_se3(1, 3) = pose.y;
    return frame;
  }

  std::istream& stream_;
  bool header_checked_{false};
  bool exhausted_{false};
  std::optional<DetectionFrame> pending_;
};

}  // namespace prediction
