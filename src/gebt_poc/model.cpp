#include "src/gebt_poc/model.h"

namespace openturbine::gebt_poc {

Kokkos::View<Section*> CreateSections(std::vector<Section> sections) {
    auto sections_view = Kokkos::View<Section*>("sections", sections.size());
    auto sections_host = Kokkos::create_mirror(sections_view);

    for (std::size_t index = 0; index < sections.size(); ++index) {
        sections_host(index) = sections[index];
    }

    // *** Question for David: How to implement a deep_copy() for a Kokkos::View<Section*>?
    Kokkos::parallel_for(
        sections.size(),
        KOKKOS_LAMBDA(std::size_t index) { sections_view(index) = sections_host(index); }
    );

    return sections_view;
}

Model::Model(std::string name, Kokkos::View<Section*> sections) : name_(name), sections_(sections) {
}

}  // namespace openturbine::gebt_poc
