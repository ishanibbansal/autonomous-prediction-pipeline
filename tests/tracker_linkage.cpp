#include "prediction/tracker.hpp"

namespace prediction {
namespace test_support {

/// Touched from the test binary so this translation unit is linked. The header
/// is included here and in the test TU, which is the ODR check for a header-only
/// tracker with inline methods and static data members.
const MultiObjectTracker& linkage_anchor() {
  static const MultiObjectTracker tracker;
  return tracker;
}

}  // namespace test_support
}  // namespace prediction
