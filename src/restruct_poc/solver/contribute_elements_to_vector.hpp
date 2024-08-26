#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {
struct ContributeElementsToVector {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<double**[6]>::const_type elements;
    Kokkos::View<double*> vector;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_nodes), [&](size_t i_node) {
            const auto node_start = node_state_indices(i_elem, i_node) * 6U;
            for (auto j = 0U; j < 6U; ++j) {
                 vector(node_start + j) = elements(i_elem, i_node, j);
            }
        });
    }
};
}  // namespace openturbine
