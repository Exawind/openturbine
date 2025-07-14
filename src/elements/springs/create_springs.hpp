#pragma once

#include "model/node.hpp"
#include "springs.hpp"
#include "springs_input.hpp"

namespace openturbine {

template <typename DeviceType>
inline Springs<DeviceType> CreateSprings(
    const SpringsInput& springs_input, const std::vector<Node>& nodes
) {
    Springs<DeviceType> springs(springs_input.NumElements());

    auto host_node_state_indices =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, springs.node_state_indices);
    auto host_x0 = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, springs.x0);
    auto host_l_ref = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, springs.l_ref);
    auto host_k = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, springs.k);

    for (auto elementIdx = 0U; elementIdx < springs_input.NumElements(); elementIdx++) {
        const auto& element = springs_input.elements[elementIdx];

        host_node_state_indices(elementIdx, 0U) = static_cast<size_t>(element.node_ids[0]);
        host_node_state_indices(elementIdx, 1U) = static_cast<size_t>(element.node_ids[1]);

        for (auto component = 0U; component < 3U; component++) {
            host_x0(elementIdx, component) =
                nodes[element.node_ids[1]].x0[component] - nodes[element.node_ids[0]].x0[component];
        }

        host_l_ref(elementIdx) = element.undeformed_length;
        host_k(elementIdx) = element.stiffness;
    }

    Kokkos::deep_copy(springs.node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(springs.x0, host_x0);
    Kokkos::deep_copy(springs.l_ref, host_l_ref);
    Kokkos::deep_copy(springs.k, host_k);

    return springs;
}

}  // namespace openturbine
