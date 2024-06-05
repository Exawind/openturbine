#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct MassesSetNodeStateIndices {
    Kokkos::View<int*> node_state_indices;
    int start_index;

    KOKKOS_FUNCTION
    void operator()(int i) const { node_state_indices(i) = start_index + i; }
};

}  // namespace openturbine
