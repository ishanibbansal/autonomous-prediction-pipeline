# Autonomous Prediction Pipeline

A standalone C++17 library for multi-object tracking and trajectory prediction. This repository currently contains Milestone 1 through Phase 2: detection contracts, ego-to-world coordinate transforms, and stream-based CSV ingestion. It does not yet include a tracker, a motion model, or a predictor.

Detections can be transformed with a planar `SE(2)` pose or projected from an upstream `4 x 4` `SE(3)` ego-to-world transform onto the global bird's-eye-view plane. Downstream track and trajectory structs are declared so later phases can link against the same contracts.

## Phase 1

- Header-only `prediction_lib` with `Pose2D`, `Detection`, `DetectionFrame`, and the tracking/prediction contracts.
- Angle wrapping into `[-pi, pi]`.
- `SE(2)` ego/world transforms and their inverse.
- `SE(3)` projection that drops world `z` and takes heading from the rotated local forward vector.
- GoogleTest coverage for wrapping, `SE(2)` round trips, planar `SE(3)` equivalence, and a pitched/rolled BEV projection.

## Phase 2

`prediction::CsvReader` reads an unquoted perception CSV from any `std::istream` and groups consecutive valid rows with the same `frame_id` into `DetectionFrame` packets.

```text
frame_id,timestamp,ego_x,ego_y,ego_yaw,det_x,det_y,size_x,size_y,heading,score
```

- `read_next_frame(DetectionFrame&)` streams one frame at a time. `load_all_frames` is a convenience wrapper. On `false`, the output frame is unchanged.
- Frame IDs must be digit-only `uint64_t` values. All other required numeric fields must parse completely and be finite. Detection headings are normalized, and nonpositive box dimensions are rejected.
- Each valid frame receives both a planar ego pose and a matching roll- and pitch-free `SE(3)` ego-to-world transform.
- An empty frame is represented by one row whose six detection fields are individually empty or case-insensitive `nan`. Partial sentinels and other malformed rows are skipped without interrupting valid rows around them.
- The first valid row defines a frame's timestamp and ego pose. Later rows for that frame with conflicting metadata are skipped.

Engineering decisions and verification notes live in [`docs/dev_log.md`](docs/dev_log.md).

## Directory structure

```
├── CMakeLists.txt                 # C++17, Eigen3, warning flags on prediction_lib
├── include/prediction/
│   ├── types.hpp                 # Contracts and SE(2) / SE(3) transforms
│   └── csv_reader.hpp            # Stream-based CSV ingestion and frame grouping
├── tests/
│   ├── CMakeLists.txt             # GoogleTest v1.14.0 via FetchContent
│   ├── test_ingestion.cpp         # Angle wrap and frame-transform tests
│   └── test_csv_reader.cpp        # CSV schema, grouping, and malformed-input tests
└── docs/dev_log.md               # Engineering scratchpad, not a user guide
```

## Build and test

Requires CMake 3.16+, a C++17 compiler, and an installed Eigen3. GoogleTest v1.14.0 is fetched at configure time, so the first configure needs network access unless that tag is already in the CMake fetch cache.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Warning flags (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`, or `/W4` on MSVC) are attached to `prediction_lib` and therefore to project tests. They are not applied globally, so the fetched GoogleTest sources are not rebuilt under those flags.
