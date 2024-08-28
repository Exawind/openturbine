#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {
struct ComputeNumberOfNonZeros {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;

    KOKKOS_FUNCTION
    void operator()(int i_elem, size_t& update) const {
        const auto num_nodes = num_nodes_per_element(i_elem);
        update += (num_nodes * 6U) * (num_nodes * 6U);
    }
};
}  // namespace openturbine
