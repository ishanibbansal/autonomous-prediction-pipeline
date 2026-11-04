# Engineering Log & Scratchpad

This file is a living technical journal for `autonomous-prediction-pipeline`. It records architectural decisions, verification, and limits of the code that actually landed. It is not a user guide.

## Milestone 1 — Detection contracts and coordinate frames

### Log 1.1: Phase 1 header contracts and BEV projection

- **Date:** October 30, 2026
- **Objective:** Stand up a C++17 header library that can take an ego-frame detection into the world BEV plane from either a planar pose or an upstream `4 x 4` rigid transform, plus the downstream track and trajectory structs later phases will fill in.
- **What landed:**
  - `prediction_lib` is an interface library. Phase 1 has no translation units. Include path is `include/`, and the only link dependency is `Eigen3::Eigen`.
  - Warning flags live on `prediction_lib` (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`, or `/W4` on MSVC). They are not global `add_compile_options`, so fetched GoogleTest is not rebuilt under `-Wconversion`.
  - `kPi` / `kTwoPi` are `inline constexpr` in `namespace prediction`. C++17 has no `std::numbers::pi`, and `M_PI` is not standard.
  - `normalize_angle` is `atan2(sin(angle), cos(angle))`. For a finite angle this lands in `[-pi, pi]`. `+pi` and `-pi` are the same heading. Odd multiples of pi can come back as either endpoint because `sin(pi)` is a tiny residual whose sign is not stable.
  - `SE(2)` ego-to-world and world-to-ego follow the spec rotation. The pose argument is always ego-to-world, including on the inverse. Size and score are copied.
  - `SE(3)` treats the detection as a point on the ego `z = 0` plane, applies the `4 x 4`, and keeps only world `x` and `y`. Heading is `atan2` of the transformed local forward vector `[cos(psi), sin(psi), 0, 0]`, then wrapped. It is not `psi_ego + yaw(T)`. World `z` is discarded. Pitched box extents are not recomputed.
- **Rotation convention used by the pitch/roll test:** active intrinsic `R = Rz(yaw) * Ry(pitch) * Rx(roll)`, which is the same matrix as extrinsic `X` then `Y` then `Z`. The test rebuilds the expected point and forward vector from that matrix instead of trusting the function under test. The specified case uses ego heading `0`. A second assertion uses ego heading `0.35` so a wrong implementation that only adds the transform yaw cannot pass.
- **Boundary chosen for the angle test:** odd multiples of pi are checked with `abs(abs(normalized) - pi) < 1e-9`, not with a required sign. Other angles (`0`, `2 pi`, `5.5 pi -> -0.5 pi`, `-7.25 pi -> 0.75 pi`) are checked with `EXPECT_NEAR` plus sine/cosine equivalence.

### Log 1.2: Configure, build, and ctest

- **Date:** October 30, 2026
- **Commands, from the repository root:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

- **Observed:**
  - Compiler: GNU C++ 15.2.0. Eigen headers resolved from `/usr/include/eigen3` and were passed as a system include, so Eigen itself did not emit conversion warnings into `test_ingestion.cpp`.
  - GoogleTest v1.14.0 was consumed through `FetchContent` and built under its own flags, not the project warning set. The project compile line for `tests/test_ingestion.cpp` was `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -std=c++17`. That translation unit produced no warnings.
  - `ctest` result: 4/4 passed (`AngleNormalizationTest`, `SE2Transform_RoundTripAndInvariance`, `SE3Transform_EquivalenceWithSE2`, `SE3Transform_PitchRollProjection`). Total test time 0.01 s. Zero failures.

### Log 1.3: Limits left in this phase

- **Date:** October 30, 2026
- `normalize_angle` is specified only for finite inputs. NaN and infinity are not mapped to a heading.
- If a transformed forward vector is exactly vertical, its BEV heading is undefined (`atan2(0, 0)`). The 10 degree roll / 5 degree pitch case does not hit that singularity, and the function does not special-case it.
- There is no `SE(3)` world-to-ego inverse. Only the planar inverse is implemented.
- `DetectionFrame` stores both `ego_pose_2d` and `ego_pose_se3`. Nothing checks that one is the planar projection of the other. Callers pick which transform to apply.
- Track and trajectory structs were contracts only at the end of Milestone 1. The single-target filter landed in Milestone 2 Phase 1. Gated association landed in Milestone 2 Phase 2. A multi-object tracker and trajectory predictor are still absent.

### Log 1.4: Phase 2 stream-based CSV ingestion and frame grouping

- **Date:** October 31, 2026
- **What landed:** Header-only `prediction::CsvReader(std::istream&)`, `read_next_frame`, and `load_all_frames`. No new dependencies or library target changes.
- **CSV contract:** The first nonblank line must contain the eleven schema names in order; names and numeric tokens are whitespace-trimmed. Quoted CSV fields are not supported. Invalid/missing headers permanently exhaust the reader. Input rows are assumed ordered by frame ID; grouping is consecutive, not a global merge or sort.
- **Validation:** Frame IDs are digit-only uint64 values with overflow checks. Floating-point tokens must parse completely and be finite. Malformed rows are skipped, including partial detections, trailing numeric garbage, and nonpositive box dimensions. Detection headings are normalized; ego yaw is retained as supplied. Score is retained without an additional range restriction.
- **Empty frames:** All six detection fields must individually be empty or case-insensitive `nan` to form a sentinel. Mixed empty/NaN sentinels are supported; partial sentinels are rejected. Sentinels never erase accumulated detections.
- **Grouping and poses:** Only valid rows start frames or trigger boundaries. One validated lookahead row is cached; the last frame flushes exactly once at EOF. Repeated-ID metadata must exactly equal the first valid row's timestamp and pose, otherwise the row is skipped. Each frame has consistent planar 2D and SE(3) ego-to-world poses. A false return leaves the output unchanged.
- **Tests:** Added `test_csv_reader` with 11 discovered cases covering grouping, streaming/load-all equivalence, independent SE(3) expectations, heading normalization, malformed-row recovery, full uint64 range, sentinel variants and every partial sentinel combination, metadata conflicts, and unchanged output on failure. Computed poses/angles use tolerances against independent expectations.
- **Verification:** Ran a clean rebuild followed by `cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build && ctest --test-dir build --output-on-failure`. All **15/15 tests passed** (4 Phase 1, 11 Phase 2), with zero compiler warnings. Project targets retain `-std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow` and compiler extensions remain disabled.

## Milestone 2 — Single-target motion model

### Log 2.1: Phase 1 6D Kalman filter

- **Date:** November 2, 2026
- **What landed:** Header-only `prediction::KalmanFilter` and `KalmanFilterConfig` in `include/prediction/kalman_filter.hpp`. No new dependencies. `tests/kalman_filter_linkage.cpp` includes the same header from a second translation unit so an ODR break would fail the link of `test_kalman_filter`.
- **State and model:** State remains the Milestone 1 contract `[x, y, v_x, v_y, psi, omega]`. Predict is constant velocity plus constant turn rate. Those blocks are decoupled on purpose. A coordinated-turn model would rotate `(v_x, v_y)` by `omega * dt`; this one does not. Heading is directed and `2 pi` periodic. Box orientation with a `pi` ambiguity is not handled here.
- **Process noise name:** The specified `(1/4) dt^4`, `(1/2) dt^3`, `dt^2` blocks are discrete piecewise-constant white acceleration noise, with variances in `m^2/s^4` and `rad^2/s^4`. They are not the continuous-time white-noise integrals `(1/3) dt^3`, `(1/2) dt^2`, `dt`. Subdividing a predict interval changes the accumulated `Q`. For `dt > 0`, each of the three noise blocks has rank 1, so `Q` has three positive eigenvalues and three exact zeros.
- **Failure contract:** The constructor throws `std::invalid_argument` unless all eight variances are finite and `> 0`. `initialize` returns `std::nullopt` for a detection that fails the shared checks: finite position, heading, and score, and both size components finite and `> 0`. Score is not clipped to `[0, 1]`, matching `CsvReader`. `predict` and `update` return `false` without writing the track when `dt` is non-finite or negative, the track is not admissible, the detection is invalid, an LDLT of `S` is not strictly positive definite, or a computed matrix is non-finite. `dt == 0` on an admissible track returns `true` and does not rewrite state, covariance, size, id, or status, including an unnormalized heading.
- **Admissibility:** State, covariance, and size must be finite, both size components must be `> 0`, and covariance must be symmetric to `1e-9 * (1 + max(|Pij|, |Pji|))`. Positive-semidefinite input is checked with `SelfAdjointEigenSolver`: the smallest eigenvalue may be negative only by `1e-9 * (1 + max(0, lambda_max))`. A singular covariance is allowed. That check is not used as a substitute for a positive-definite `S`.
- **Update numerics:** `S` must report `LDLT` success, `isPositive()`, finite `vectorD()`, and every pivot strictly greater than zero. The gain is `ldlt.solve((P H^T).transpose()).transpose()`. The covariance write is the Joseph form, then `P = 0.5 (P + P^T)`. Squared Mahalanobis distances in `[-1e-12, 0)` are clamped to zero; anything more negative or non-finite is rejected. `update` does not apply a gate, and neither `predict` nor `update` writes `track_id` or `status`.
- **Caller obligations:** Measurements are assumed world-frame. Nothing in `Detection` can prove that. Heading evolution between updates must not be aliased by more than `pi`, or the wrapped residual will pull turn rate the wrong way.
- **Tests:** Initialization and config rejection; independent `F`/`Q` propagation; velocity and turn-rate convergence over 40 steps; predict and update crossing the `+/- pi` cut, with the update prior loose enough that the posterior actually crosses; covariance shrinkage, symmetry, and positive eigenvalues on a well-conditioned birth update; Mahalanobis agreement and `+2 pi` invariance; rejection of bad `dt`, bad detections, asymmetric covariance, and an indefinite covariance without mutation; second-TU linkage.
- **Verification:** `cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build && ctest --test-dir build --output-on-failure` on November 2, 2026. **25/25 passed** (4 Phase 1, 11 Phase 2, 10 Kalman). The Kalman objects were compiled as `-std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow` with extensions off, and that rebuild produced no warnings. A first build failed on a transposed `P H^T` (6x3 written into a 3x6 temporary); the gain path is now `solve((P H^T)^T)^T`, which is the shape `LDLT` expects.

### Log 2.2: Phase 2 gated Hungarian association

- **Date:** November 3, 2026
- **What landed:** Header-only `prediction::DataAssociation`, `DataAssociationConfig`, `TrackDetectionMatch`, `AssociationResult`, and `CostMatrix` in `include/prediction/data_association.hpp`. No new dependencies. `tests/data_association_linkage.cpp` includes the same header from a second translation unit.
- **Objective:** Maximum cardinality over gate-admissible pairs, then minimum sum of squared Mahalanobis distances. The solver is a rectangular shortest-augmenting-path Hungarian reduction, not a permutation search. Complexity is `O((M + N)^3)` arithmetic steps with `O((M + N)^2)` auxiliary storage. Cost construction is a separate `O(MN)` pass because the filter state dimension is fixed.
- **Normalization:** A threshold merely larger than `G` does not guarantee maximum cardinality. An augmenting path can exchange several cheap matches for one more expensive match. Every admissible cost is therefore divided by `G`, placing it in `[0, 1]`. Each of the `M` tracks receives one private dummy column at cost `P = min(M, N) + 1`. A matching with `k` real pairs then costs somewhere in `[(M - k) P, (M - k) P + min(M, N)]`. Those intervals do not overlap, so gaining one match beats every possible reduction in the real-cost sum. The reported `mahalanobis_distance_sq` remains the unnormalized filter value. This also avoids forming a penalty such as `(K + 1) G`, which can overflow for a large threshold.
- **Forbidden edges:** A pair whose distance is `nullopt`, non-finite, or strictly greater than `G` is absent from the augmenting-path search. It is not encoded as a large finite number. The scan never lets a forbidden edge erase or replace the slack of another column. Zero-size default tracks and detections inherit the filter's existing rejection and stay unmatched. An asymmetric or indefinite covariance does the same. Valid pairs in the same call are still assigned, and neither input is mutated.
- **Gate and ties:** Acceptance is `d2 <= G` with no epsilon. Equality is deliberately inside the gate. Equal-cost assignments are deterministic for a fixed input because the Dijkstra frontier and column scans always advance in increasing index order. That tie break is not documented as the lexicographically smallest matching; it is only required to be repeatable.
- **Empty shapes:** `CostMatrix` stores `track_count` and `detection_count` independently of the row-major cost vector, so `0 x N` and `M x 0` remain distinguishable. Both short-circuit before the solver. `solve_cost_matrix` rejects a size mismatch and any present cost that is negative or non-finite.
- **Caller obligations:** Association does not call `predict`, does not transform detections, and does not inspect `TrackStatus`. A caller that wants association at a future time must predict first and pass the predicted tracks. Measurements must already be world-frame.
- **Tests:** Invalid thresholds; a 2x2 anti-greedy reversal; a 3x3 case where three matches require giving up two cheap edges; exact gate equality and a value just outside; `M < N`, `M > N`, both empty shapes, and a fully gated-out pair; malformed tracks and detections beside one valid pair; wrapped-heading agreement with the filter; repeatable equal-cost assignment; direct matrix validation; second-TU linkage.
- **Verification:** `cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build && ctest --test-dir build --output-on-failure` on November 3, 2026. **35/35 passed**, including the 10 new association tests. The data-association objects compiled as `-std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow` with extensions off. The build emitted no warnings.
