#include "src/gebt_poc/model.h"

namespace openturbine::gebt_poc {

Model::Model(std::string name, Kokkos::View<Section*> sections)
    : name_(name), sections_("sections", sections.extent(0)) {
    Kokkos::deep_copy(sections_, sections);
}

}  // namespace openturbine::gebt_poc
