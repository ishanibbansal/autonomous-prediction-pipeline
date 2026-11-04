#include "prediction/data_association.hpp"

namespace prediction {
namespace test_support {

/// Touched from the test binary so this translation unit is linked. The header
/// is included here and in the test TU, which is the ODR check for the
/// header-only associator.
const DataAssociation& data_association_linkage_anchor() {
  static const DataAssociation associator;
  return associator;
}

}  // namespace test_support
}  // namespace prediction
