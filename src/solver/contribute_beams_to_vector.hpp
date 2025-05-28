#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct ContributeBeamsToVector {
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t** [6], DeviceType>::const_type element_freedom_table;
    typename Kokkos::View<double** [6], DeviceType>::const_type elements;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> vector;

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        const auto element = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(element);
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;

        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, num_nodes*6U), [&](size_t node_component) {
            const auto node = node_component % num_nodes;
            const auto component = node_component / num_nodes;
            const auto entry = element_freedom_table(element, node, component);
            if constexpr (force_atomic) {
                Kokkos::atomic_add(
                    &vector(entry, 0),
                    elements(element, node, component)
                );
            } else {
                vector(entry, 0) +=
                    elements(element, node, component);
            }
        });
    }
};

}  // namespace openturbine
