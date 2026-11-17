#include "prediction/metrics.hpp"

namespace prediction {
namespace test_support {

/// Touched from the test binary so this translation unit is linked. The header
/// is included here and in the test TU, which is the ODR check for the
/// header-only evaluator.
const TrajectoryEvaluator& metrics_linkage_anchor() {
  static const TrajectoryEvaluator evaluator;
  return evaluator;
}

}  // namespace test_support
}  // namespace prediction
