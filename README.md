# Autonomous Prediction Pipeline

A standalone C++17 library for multi-object tracking and trajectory prediction. This repository currently contains Milestone 1, Milestone 2, and Milestone 3 Phase 1: detection contracts, ego-to-world coordinate transforms, stream-based CSV ingestion, a single-target 6D Kalman filter, gated global data association, a multi-object tracker, and a kinematic multimodal trajectory predictor.

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

## Milestone 2, Phase 1

`prediction::KalmanFilter` is a header-only single-target filter on `TrackState`. The world-frame state is `[x, y, v_x, v_y, psi, omega]`. Measurements are world-frame detections `[x, y, heading]`. Heading is a directed `2 pi` angle, and the yaw residual is wrapped into `[-pi, pi]`.

- Process noise is discrete piecewise-constant white acceleration noise. The translational and yaw blocks use `(1/4) dt^4`, `(1/2) dt^3`, and `dt^2` times the configured acceleration variance. Splitting one interval into smaller predict steps changes the accumulated noise.
- `(v_x, v_y)` and `(psi, omega)` are independent linear blocks. This is not a coordinated-turn model: turn rate does not rotate the velocity vector.
- `predict` and `update` never change `track_id` or `status`. `update` does not gate. A large but valid innovation is applied; association belongs to the caller.
- The caller must pass world-frame detections. `Detection` does not record its frame, so the filter cannot check that itself.
- Configuration variances must be finite and strictly positive. An invalid config throws `std::invalid_argument` from the constructor. The stored copy is available only through `config()`.
- `initialize` returns `std::nullopt` for a detection whose position, heading, or score is non-finite, or whose length or width is not strictly positive. Score is not restricted to `[0, 1]`.
- `predict`, `update`, and `mahalanobis_distance_squared` fail transactionally. A negative, NaN, or infinite `dt`, a non-admissible track, a malformed detection, a failed or non-positive-definite innovation covariance, or a non-finite computed result leaves the track unchanged. `predict` with `dt == 0` on an admissible track is an exact no-op.
- Innovation systems are solved with `Eigen::LDLT`, not an explicit inverse. Covariance is symmetrized after propagation and update. The measurement update uses the Joseph form.

## Milestone 2, Phase 2

`prediction::DataAssociation` solves gated track-to-detection assignment. It does not predict or update tracks. Both the track state and every detection must already be in the same world frame.

- A pair is admissible when `KalmanFilter::mahalanobis_distance_squared` returns a finite value no greater than `gate_threshold_sq`. The default threshold is the 3-degree chi-squared 0.99 bound, `11.344866730144373`. The comparison is exact: equality is accepted.
- `gate_threshold_sq` must be finite and strictly positive. An invalid config throws `std::invalid_argument` from the constructor.
- Among admissible one-to-one matchings, the result first maximizes the number of matches and then minimizes the sum of squared Mahalanobis distances. Admissible solver costs are `d^2 / gate_threshold_sq`, so they lie in `[0, 1]`. Each track also has one private dummy column with cost `min(M, N) + 1`. Forbidden pairs, including any pair for which the filter returns `std::nullopt`, are omitted from the search rather than represented by a large finite cost.
- A non-admissible track or malformed detection therefore stays unmatched and does not disturb any other admissible pair. Inputs are not mutated.
- `matches` are sorted by ascending track index. `unmatched_tracks` and `unmatched_detections` are sorted ascending. Every supplied index appears in exactly one of those collections.
- `solve_cost_matrix` exposes the same rectangular assignment solver for direct tests. It preserves `0 x N` and `M x 0` dimensions and adds the cardinality-safe dummy penalty internally. Present costs must be finite and nonnegative, and should be normalized to at most 1 when cardinality priority is required.

## Multi-object tracker

`prediction::MultiObjectTracker` owns the track list. Each frame is transformed into the world frame, predicted once, associated, updated or coasted, pruned, then used to birth new tracks. Track state stays the Milestone 1 `TrackState`. Hit counters and timestamps live in a parallel `TrackRecord`, so association still receives the track vector directly.

Lifecycle, in order:

- A birth is `Tentative`, with one hit. If `min_hits_to_confirm` is 1, it is `Confirmed` in that same frame.
- Further consecutive hits promote `Tentative` to `Confirmed` once `consecutive_hits` reaches `min_hits_to_confirm`.
- A miss clears the consecutive-hit counter. `Confirmed` becomes `Lost`. A later hit returns `Lost` to `Confirmed` without waiting for another confirmation streak. `Lost` does not become `Tentative`.
- A `Tentative` track is removed when `consecutive_misses` exceeds `max_tentative_misses`. The default of 0 drops it on the first miss. Any status is removed when `consecutive_misses` exceeds `max_consecutive_misses`.

`TrackerConfig` defaults:

- `min_hits_to_confirm = 3`
- `max_consecutive_misses = 5`
- `max_tentative_misses = 0`
- `output_policy = ConfirmedAndLost`
- `pose_source = Planar`

Those three counters are checked before the filter and associator are built. `0` confirmation hits, `0` allowed misses, or a tentative-miss limit above the general miss limit throws `std::invalid_argument` naming the field. Invalid nested filter or association settings still throw from those constructors.

Output policies are `ConfirmedOnly`, `ConfirmedAndLost`, and `All`. `process_frame` returns a reference to the policy view of the live tracks. That reference is invalidated by the next `process_frame` or `reset`. `tracks(policy)` returns a copy. Live tracks, including those hidden by the policy, stay in ascending `track_id` order, starting at 1. `reset()` clears them and starts again at 1. Every returned state has been predicted to the latest accepted timestamp.

`PoseSource::Planar` uses `ego_pose_2d`. `PoseSource::SE3` uses `ego_pose_se3`. The other pose is not read and may be non-finite.

`process_frame` rejects a frame before it changes any tracker state when the timestamp is non-finite, the selected pose is non-finite, `frame_id` is not strictly greater than the previous accepted id, or the timestamp moves backwards. The first frame after construction or `reset()` may use any id, including 0, and any finite timestamp. Equal timestamps are accepted and produce `dt = 0`, which is still a hit if the detection matches. A rejected frame throws `std::invalid_argument` and leaves tracks, records, ids, history, and the previous output buffer unchanged.

```cpp
std::ifstream file("perception.csv");
prediction::CsvReader reader(file);
prediction::MultiObjectTracker tracker;
prediction::DetectionFrame frame;
while (reader.read_next_frame(frame)) {
  const std::vector<prediction::TrackState>& tracks = tracker.process_frame(frame);
  // tracks is invalidated by the next process_frame or reset.
}
```

## Milestone 3, Phase 1

`prediction::KinematicTrajectoryPredictor` forecasts one world-frame `TrackState` as three constant-speed modes. It does not track, associate, or advance a coasted track. The caller supplies a state that is already valid at `current_timestamp`.

- Samples are future steps only. With `N = round(H / dt)`, sample `m` is at `current_timestamp + m * dt`. The realized horizon is `N * dt`, which can overshoot or undershoot the requested horizon. `N` must be in `[1, 10000]`.
- World velocity stays independent of body heading, matching the Kalman filter. A mode with rate `r` rotates the supplied velocity. It does not rebuild velocity as speed along heading. Positive `r` is counterclockwise in the world frame, not an overtake; negative `r` is clockwise, not a yield.
- The straight rate is `0`. The other rates are `max(omega, branch_turn_rate)` and `min(omega, -branch_turn_rate)`. Those are minimum signed rates, not `omega +/- branch_turn_rate`.
- The circular displacement uses `t * sinc(r t / 2) * R(r t / 2)`. Near-zero angles use a Taylor series, so a tiny rate does not divide by the rate or subtract nearly equal cosines.
- An exactly stationary track holds its position and normalized heading in every mode. Any nonzero finite velocity is rolled out. There is no 0.1 m/s snap to zero.
- `modes` is the authoritative result, ordered straight, counterclockwise, clockwise. `waypoints` remains the Milestone 1 field and copies the straight mode without velocity. `current_state` is an unchanged copy of the input, including an unnormalized heading.
- Position covariance is the same heuristic in every mode: `P_pp + t^2 P_vv + q t I`. It ignores cross covariance and angular uncertainty. It is symmetric and positive semidefinite, not necessarily positive definite, and its diagonals need not grow when velocity variance and diffusion are zero. `q` is `pos_diffusion_rate` in m^2/s and is not squared again.
- Mixture weights are normalized when they already sum to 1 within `1e-6`. They are heuristic weights, not calibrated intent probabilities.
- Configuration and malformed-track failures throw `std::invalid_argument` and name the field. A nonfinite computed sample, or a finite timestamp grid that is no longer strictly increasing, throws `std::overflow_error`. A batch rejects the whole call; it does not return a prefix. Status is not filtered.
- The returned vectors may allocate on every call. There is no zero-allocation workspace overload in this phase.

Engineering decisions and verification notes live in [`docs/dev_log.md`](docs/dev_log.md).

## Directory structure

```
├── CMakeLists.txt                 # C++17, Eigen3, warning flags on prediction_lib
├── include/prediction/
│   ├── types.hpp                 # Contracts and SE(2) / SE(3) transforms
│   ├── csv_reader.hpp            # Stream-based CSV ingestion and frame grouping
│   ├── kalman_filter.hpp         # Single-target 6D world-frame Kalman filter
│   ├── data_association.hpp      # Gated Hungarian track-to-detection assignment
│   ├── tracker.hpp               # Multi-object lifecycle: confirm, coast, prune, birth
│   └── predictor.hpp             # Constant-speed three-mode trajectory forecast
├── tests/
│   ├── CMakeLists.txt                # GoogleTest v1.14.0 via FetchContent
│   ├── test_ingestion.cpp            # Angle wrap and frame-transform tests
│   ├── test_csv_reader.cpp           # CSV schema, grouping, and malformed-input tests
│   ├── test_kalman_filter.cpp        # Filter contract, wrap, and rejection tests
│   ├── kalman_filter_linkage.cpp     # Second TU for the header-only filter ODR check
│   ├── test_data_association.cpp     # Gating, optimality, and rectangular assignment tests
│   ├── data_association_linkage.cpp  # Second TU for the header-only associator ODR check
│   ├── test_tracker.cpp              # Lifecycle, coasting, validation, and crossing tests
│   ├── tracker_linkage.cpp           # Second TU for the header-only tracker ODR check
│   ├── test_predictor.cpp            # Kinematic rollout, covariance, and rejection tests
│   └── predictor_linkage.cpp         # Second TU for the header-only predictor ODR check
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
