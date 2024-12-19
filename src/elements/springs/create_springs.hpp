#pragma once

#include "springs.hpp"
#include "springs_input.hpp"

namespace openturbine {

inline Springs CreateSprings(const SpringsInput& springs_input) {
    Springs springs(springs_input.NumElements());

    auto host_node_state_indices = Kokkos::create_mirror(springs.node_state_indices);
    auto host_x0 = Kokkos::create_mirror(springs.x0);
    auto host_l_ref = Kokkos::create_mirror(springs.l_ref);
    auto host_k = Kokkos::create_mirror(springs.k);

    for (size_t i_elem = 0; i_elem < springs_input.NumElements(); i_elem++) {
        const auto& element = springs_input.elements[i_elem];

        host_node_state_indices(i_elem, 0U) = static_cast<size_t>(element.nodes[0].ID);
        host_node_state_indices(i_elem, 1U) = static_cast<size_t>(element.nodes[1].ID);

        for (size_t i = 0; i < 3; i++) {
            host_x0(i_elem, i) = element.nodes[1].x[i] - element.nodes[0].x[i];
        }

        host_l_ref(i_elem) = element.undeformed_length;
        host_k(i_elem) = element.stiffness;
    }

    Kokkos::deep_copy(springs.node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(springs.x0, host_x0);
    Kokkos::deep_copy(springs.l_ref, host_l_ref);
    Kokkos::deep_copy(springs.k, host_k);

    return springs;
}

}  // namespace openturbine
