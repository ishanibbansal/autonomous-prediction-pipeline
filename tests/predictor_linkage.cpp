#include "prediction/predictor.hpp"

namespace prediction {
namespace test_support {

/// Touched from the test binary so this translation unit is linked. The header
/// is included here and in the test TU, which is the ODR check for the
/// header-only predictor.
const KinematicTrajectoryPredictor& predictor_linkage_anchor() {
  static const KinematicTrajectoryPredictor predictor;
  return predictor;
}

}  // namespace test_support
}  // namespace prediction
