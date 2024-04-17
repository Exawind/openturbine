#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct SetNodeStateIndices {
    Kokkos::View<int*> node_state_indices;
    KOKKOS_FUNCTION
    void operator()(int i) const { node_state_indices(i) = i; }
};

}
