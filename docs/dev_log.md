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
- Track and trajectory structs are contracts only. No filter, association, or predictor exists yet.
