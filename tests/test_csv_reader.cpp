#include "prediction/csv_reader.hpp"
#include "prediction/types.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr double kTolerance = 1e-9;
const std::string kHeader =
    "frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,score\n";
using Row = std::array<std::string, 11>;

Row valid_row() {
  return {"7", "1.25", "3.5", "-2.25", "0.4", "8", "-3", "4.2", "1.7", "1.2", "0.61"};
}

std::string line(const Row& row) {
  std::string result;
  for (std::size_t i = 0; i < row.size(); ++i) {
    if (i != 0) result += ',';
    result += row[i];
  }
  return result + '\n';
}

void expect_frame_equal(const prediction::DetectionFrame& actual,
                        const prediction::DetectionFrame& expected) {
  EXPECT_EQ(actual.frame_id, expected.frame_id);
  EXPECT_DOUBLE_EQ(actual.timestamp, expected.timestamp);
  EXPECT_DOUBLE_EQ(actual.ego_pose_2d.x, expected.ego_pose_2d.x);
  EXPECT_DOUBLE_EQ(actual.ego_pose_2d.y, expected.ego_pose_2d.y);
  EXPECT_DOUBLE_EQ(actual.ego_pose_2d.yaw, expected.ego_pose_2d.yaw);
  for (Eigen::Index r = 0; r < 4; ++r) {
    for (Eigen::Index c = 0; c < 4; ++c) {
      EXPECT_DOUBLE_EQ(actual.ego_pose_se3(r, c), expected.ego_pose_se3(r, c));
    }
  }
  ASSERT_EQ(actual.detections.size(), expected.detections.size());
  for (std::size_t i = 0; i < actual.detections.size(); ++i) {
    EXPECT_DOUBLE_EQ(actual.detections[i].position.x(), expected.detections[i].position.x());
    EXPECT_DOUBLE_EQ(actual.detections[i].position.y(), expected.detections[i].position.y());
    EXPECT_DOUBLE_EQ(actual.detections[i].size.x(), expected.detections[i].size.x());
    EXPECT_DOUBLE_EQ(actual.detections[i].size.y(), expected.detections[i].size.y());
    EXPECT_DOUBLE_EQ(actual.detections[i].heading, expected.detections[i].heading);
    EXPECT_DOUBLE_EQ(actual.detections[i].score, expected.detections[i].score);
  }
}

prediction::DetectionFrame marked_frame() {
  prediction::DetectionFrame frame;
  frame.frame_id = 999;
  frame.timestamp = -12.5;
  frame.ego_pose_2d = {91.0, -17.0, 0.75};
  frame.ego_pose_se3.setConstant(42.0);
  prediction::Detection detection;
  detection.position = Eigen::Vector2d(11.0, -8.0);
  detection.size = Eigen::Vector2d(6.0, 2.0);
  detection.heading = -0.8;
  detection.score = 0.3;
  frame.detections.push_back(detection);
  return frame;
}

void expect_exhausted(prediction::CsvReader& reader, prediction::DetectionFrame& frame) {
  const prediction::DetectionFrame before = frame;
  for (int attempt = 0; attempt < 3; ++attempt) {
    EXPECT_FALSE(reader.read_next_frame(frame));
    expect_frame_equal(frame, before);
  }
}

}  // namespace

TEST(CsvReader, GroupsConsecutiveValidIdsWithLookaheadAndFinalFlush) {
  Row first = valid_row();
  Row second = first;
  second[5] = "9";
  Row next = first;
  next[0] = "8";
  next[1] = "2.5";
  std::string csv = kHeader + line(first) + "\n \t\n" + line(second);
  csv += "9,broken\n";  // An invalid intervening ID must not split the group.
  csv += line(first) + line(next) + line(first);
  csv.pop_back();  // The final frame must flush without a trailing newline.
  std::istringstream input(csv);
  prediction::CsvReader reader(input);
  prediction::DetectionFrame frame;
  ASSERT_TRUE(reader.read_next_frame(frame));
  EXPECT_EQ(frame.frame_id, 7U);
  ASSERT_EQ(frame.detections.size(), 3U);
  EXPECT_DOUBLE_EQ(frame.detections[1].position.x(), 9.0);
  ASSERT_TRUE(reader.read_next_frame(frame));
  EXPECT_EQ(frame.frame_id, 8U);
  EXPECT_DOUBLE_EQ(frame.timestamp, 2.5);
  ASSERT_EQ(frame.detections.size(), 1U);
  ASSERT_TRUE(reader.read_next_frame(frame));
  EXPECT_EQ(frame.frame_id, 7U);  // IDs are grouped, not globally merged or sorted.
  ASSERT_EQ(frame.detections.size(), 1U);
  expect_exhausted(reader, frame);
}

TEST(CsvReader, ParsesFieldsAndConstructsIndependentPlanarSE3) {
  std::istringstream input(kHeader + line(valid_row()));
  prediction::CsvReader reader(input);
  prediction::DetectionFrame frame;
  ASSERT_TRUE(reader.read_next_frame(frame));
  EXPECT_EQ(frame.frame_id, 7U);
  EXPECT_DOUBLE_EQ(frame.timestamp, 1.25);
  EXPECT_DOUBLE_EQ(frame.ego_pose_2d.x, 3.5);
  EXPECT_DOUBLE_EQ(frame.ego_pose_2d.y, -2.25);
  EXPECT_DOUBLE_EQ(frame.ego_pose_2d.yaw, 0.4);
  Eigen::Matrix4d expected;
  expected << std::cos(0.4), -std::sin(0.4), 0.0, 3.5,
              std::sin(0.4), std::cos(0.4), 0.0, -2.25,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0;
  for (Eigen::Index r = 0; r < 4; ++r) {
    for (Eigen::Index c = 0; c < 4; ++c) {
      EXPECT_NEAR(frame.ego_pose_se3(r, c), expected(r, c), kTolerance);
    }
  }
  ASSERT_EQ(frame.detections.size(), 1U);
  EXPECT_DOUBLE_EQ(frame.detections[0].position.x(), 8.0);
  EXPECT_DOUBLE_EQ(frame.detections[0].position.y(), -3.0);
  EXPECT_DOUBLE_EQ(frame.detections[0].size.x(), 4.2);
  EXPECT_DOUBLE_EQ(frame.detections[0].size.y(), 1.7);
  EXPECT_DOUBLE_EQ(frame.detections[0].heading, 1.2);
  EXPECT_DOUBLE_EQ(frame.detections[0].score, 0.61);
  expect_exhausted(reader, frame);
}

TEST(CsvReader, NormalizesDetectionHeadingWithoutChangingEgoYaw) {
  Row row = valid_row();
  row[4] = "7";
  row[9] = "8";
  std::istringstream input(kHeader + line(row));
  const auto frames = prediction::CsvReader::load_all_frames(input);
  ASSERT_EQ(frames.size(), 1U);
  EXPECT_DOUBLE_EQ(frames[0].ego_pose_2d.yaw, 7.0);
  ASSERT_EQ(frames[0].detections.size(), 1U);
  EXPECT_NEAR(frames[0].detections[0].heading, 8.0 - prediction::kTwoPi, kTolerance);
  EXPECT_NEAR(frames[0].ego_pose_se3(0, 0), std::cos(7.0), kTolerance);
  EXPECT_NEAR(frames[0].ego_pose_se3(1, 0), std::sin(7.0), kTolerance);
}

TEST(CsvReader, LoadAllMatchesStreamingForSingleAndMultipleFrames) {
  Row next = valid_row();
  next[0] = "12";
  for (const std::string& body : {line(valid_row()), line(valid_row()) + line(next) + line(next)}) {
    SCOPED_TRACE(body);
    std::istringstream stream_input(kHeader + body);
    std::istringstream bulk_input(kHeader + body);
    prediction::CsvReader reader(stream_input);
    std::vector<prediction::DetectionFrame> streamed;
    prediction::DetectionFrame frame;
    while (reader.read_next_frame(frame)) streamed.push_back(frame);
    const auto loaded = prediction::CsvReader::load_all_frames(bulk_input);
    ASSERT_EQ(streamed.size(), body == line(valid_row()) ? 1U : 2U);
    ASSERT_EQ(loaded.size(), streamed.size());
    for (std::size_t i = 0; i < loaded.size(); ++i) expect_frame_equal(loaded[i], streamed[i]);
  }
}

TEST(CsvReader, SkipsBlankLinesAndTrimsHeaderAndNumericTokens) {
  Row row = valid_row();
  for (std::string& field : row) field = " \t" + field + " \t";
  std::istringstream input(
      "\n \t\r\n  frame_id , timestamp , ego_x , ego_y , ego_yaw , det_x , det_y ,"
      " size_x , size_y , heading , score \r\n\n" + line(row));
  prediction::CsvReader reader(input);
  prediction::DetectionFrame frame;
  ASSERT_TRUE(reader.read_next_frame(frame));
  EXPECT_EQ(frame.frame_id, 7U);
  ASSERT_EQ(frame.detections.size(), 1U);
  EXPECT_DOUBLE_EQ(frame.detections[0].score, 0.61);
}

TEST(CsvReader, RejectsMalformedRowsWithoutCreatingFramesOrChangingOutput) {
  std::vector<std::string> malformed = {"garbage\n", "7,1\n"};
  std::string extra = line(valid_row());
  extra.pop_back();
  malformed.push_back(extra + ",extra\n");
  const std::vector<std::string> bad_ids = {
      "", "-1", "+1", "1.5", "1e2", "7junk", "18446744073709551616", "999999999999999999999999"};
  for (const std::string& id : bad_ids) {
    Row row = valid_row();
    row[0] = id;
    malformed.push_back(line(row));
  }
  for (std::size_t column = 1; column < 11; ++column) {
    for (const std::string& token : std::vector<std::string>{"", "garbage", "1junk", "1.2junk", "1.2.3", "nan", "NaN", "inf", "-inf", "1e9999"}) {
      Row row = valid_row();
      row[column] = token;
      malformed.push_back(line(row));
    }
  }
  for (const std::size_t column : {std::size_t{7}, std::size_t{8}}) {
    for (const std::string& token : std::vector<std::string>{"0", "-0", "-1"}) {
      Row row = valid_row();
      row[column] = token;
      malformed.push_back(line(row));
    }
  }
  for (const std::string& bad : malformed) {
    SCOPED_TRACE(bad);
    std::istringstream invalid_input(kHeader + bad);
    prediction::CsvReader invalid_reader(invalid_input);
    auto frame = marked_frame();
    expect_exhausted(invalid_reader, frame);

    std::istringstream recovery_input(kHeader + bad + line(valid_row()));
    prediction::CsvReader recovery_reader(recovery_input);
    ASSERT_TRUE(recovery_reader.read_next_frame(frame));
    EXPECT_EQ(frame.frame_id, 7U);
    ASSERT_EQ(frame.detections.size(), 1U);
    EXPECT_DOUBLE_EQ(frame.timestamp, 1.25);
    expect_exhausted(recovery_reader, frame);
  }
}

TEST(CsvReader, AcceptsFullUint64Range) {
  for (const std::string& id : std::vector<std::string>{"0", "0007", "18446744073709551615"}) {
    Row row = valid_row();
    row[0] = id;
    std::istringstream input(kHeader + line(row));
    prediction::CsvReader reader(input);
    prediction::DetectionFrame frame;
    ASSERT_TRUE(reader.read_next_frame(frame));
    const std::uint64_t expected = id == "0" ? 0U :
        (id == "0007" ? 7U : std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(frame.frame_id, expected);
  }
}

TEST(CsvReader, AcceptsMixedSentinelsAndNeverErasesDetections) {
  const std::vector<std::array<std::string, 6>> sentinels = {
      {"", "", "", "", "", ""},
      {"nan", "NAN", "NaN", "nAn", "naN", "Nan"},
      {"", " NaN ", "", "nan", "", "NAN"}};
  for (const auto& fields : sentinels) {
    Row sentinel = valid_row();
    for (std::size_t i = 0; i < fields.size(); ++i) sentinel[i + 5] = fields[i];
    SCOPED_TRACE(line(sentinel));
    std::istringstream empty_input(kHeader + line(sentinel));
    prediction::CsvReader empty_reader(empty_input);
    prediction::DetectionFrame frame;
    ASSERT_TRUE(empty_reader.read_next_frame(frame));
    EXPECT_EQ(frame.frame_id, 7U);
    EXPECT_TRUE(frame.detections.empty());
    expect_exhausted(empty_reader, frame);

    std::istringstream input(kHeader + line(sentinel) + line(valid_row()) + line(sentinel) +
                             line(valid_row()) + line(sentinel));
    prediction::CsvReader reader(input);
    ASSERT_TRUE(reader.read_next_frame(frame));
    ASSERT_EQ(frame.detections.size(), 2U);
    EXPECT_DOUBLE_EQ(frame.detections[0].position.x(), 8.0);
    EXPECT_DOUBLE_EQ(frame.detections[1].position.x(), 8.0);
    expect_exhausted(reader, frame);
  }
}

TEST(CsvReader, RejectsEveryPartialSentinelCombination) {
  for (unsigned int mask = 1; mask < 63; ++mask) {
    SCOPED_TRACE(mask);
    Row row = valid_row();
    for (std::size_t i = 0; i < 6; ++i) {
      if ((mask & (1U << i)) != 0U) row[i + 5] = i % 2 == 0 ? "" : "NaN";
    }
    std::istringstream input(kHeader + line(row));
    prediction::CsvReader reader(input);
    auto frame = marked_frame();
    expect_exhausted(reader, frame);
  }
}

TEST(CsvReader, FirstValidMetadataWinsAndEveryConflictIsSkipped) {
  for (std::size_t column = 1; column <= 4; ++column) {
    SCOPED_TRACE(column);
    Row conflicting = valid_row();
    conflicting[column] += "0000000000001";  // A tiny but exact mismatch, not a tolerance check.
    conflicting[5] = "99";
    Row invalid_first = conflicting;
    invalid_first[7] = "-1";
    Row sentinel = conflicting;
    for (std::size_t i = 5; i < 11; ++i) sentinel[i] = "";
    Row next = valid_row();
    next[0] = "8";
    std::istringstream input(kHeader + line(invalid_first) + line(valid_row()) +
                             line(conflicting) + line(sentinel) + line(valid_row()) + line(next));
    prediction::CsvReader reader(input);
    prediction::DetectionFrame frame;
    ASSERT_TRUE(reader.read_next_frame(frame));
    EXPECT_DOUBLE_EQ(frame.timestamp, 1.25);
    EXPECT_DOUBLE_EQ(frame.ego_pose_2d.x, 3.5);
    EXPECT_DOUBLE_EQ(frame.ego_pose_2d.y, -2.25);
    EXPECT_DOUBLE_EQ(frame.ego_pose_2d.yaw, 0.4);
    ASSERT_EQ(frame.detections.size(), 2U);
    EXPECT_DOUBLE_EQ(frame.detections[1].position.x(), 8.0);
    ASSERT_TRUE(reader.read_next_frame(frame));
    EXPECT_EQ(frame.frame_id, 8U);
    expect_exhausted(reader, frame);
  }
}

TEST(CsvReader, InvalidOrMissingHeaderPermanentlyFailsWithoutChangingOutput) {
  const std::vector<std::string> headers = {
      "", "\n \t\n", line(valid_row()),
      "frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading\n",
      "frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,score,extra\n",
      "timestamp,frame_id,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,score\n",
      "Frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,score\n",
      "frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,confidence\n",
      "frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,heading\n"};
  for (const std::string& header : headers) {
    SCOPED_TRACE(header);
    // Empty input really has no header; invalid headers cannot resynchronize at a later valid one.
    const std::string csv = header.empty() || header == "\n \t\n" ? header :
        header + kHeader + line(valid_row());
    std::istringstream input(csv);
    prediction::CsvReader reader(input);
    auto frame = marked_frame();
    expect_exhausted(reader, frame);
    std::istringstream bulk_input(csv);
    EXPECT_TRUE(prediction::CsvReader::load_all_frames(bulk_input).empty());
  }
  std::istringstream header_only(kHeader + "\n \t\n");
  prediction::CsvReader reader(header_only);
  auto frame = marked_frame();
  expect_exhausted(reader, frame);
}
