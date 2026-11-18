#include "prediction/visualizer.hpp"

#include <string>

namespace prediction {
namespace test_support {

/// Touched from the test binary so this translation unit is linked. The header
/// is included here and in the test TU, which is the ODR check for the
/// header-only visualizer.
const BevVisualizer& visualizer_linkage_anchor() {
  static const BevVisualizer visualizer;
  return visualizer;
}

std::string visualizer_linkage_render() {
  const BevVisualizer visualizer;
  return visualizer.render_frame(0.0, {}, {}, {}, {});
}

}  // namespace test_support
}  // namespace prediction
