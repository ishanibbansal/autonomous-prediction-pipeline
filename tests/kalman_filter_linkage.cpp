#include "prediction/kalman_filter.hpp"

namespace prediction {
namespace test_support {

/// Touched from the test binary so this translation unit is linked. The header
/// is included here and in the test TU, which is the ODR check for a header-only
/// filter with inline methods and static data members.
const KalmanFilter& linkage_anchor() {
  static const KalmanFilter filter;
  return filter;
}

}  // namespace test_support
}  // namespace prediction
