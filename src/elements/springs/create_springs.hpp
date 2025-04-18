#pragma once

#include "springs.hpp"
#include "springs_input.hpp"

namespace openturbine {

inline Springs CreateSprings(const SpringsInput& springs_input, const std::vector<Node>& nodes) {
    Springs springs(springs_input.NumElements());

    auto host_node_state_indices = Kokkos::create_mirror_view(springs.node_state_indices);
    auto host_x0 = Kokkos::create_mirror_view(springs.x0);
    auto host_l_ref = Kokkos::create_mirror_view(springs.l_ref);
    auto host_k = Kokkos::create_mirror_view(springs.k);

    for (size_t i_elem = 0; i_elem < springs_input.NumElements(); i_elem++) {
        const auto& element = springs_input.elements[i_elem];

        host_node_state_indices(i_elem, 0U) = static_cast<size_t>(element.node_ids[0]);
        host_node_state_indices(i_elem, 1U) = static_cast<size_t>(element.node_ids[1]);

        for (size_t i = 0; i < 3; i++) {
            host_x0(i_elem, i) = nodes[element.node_ids[1]].x[i] - nodes[element.node_ids[0]].x[i];
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
