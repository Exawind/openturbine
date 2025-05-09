#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct ContributeBeamsToVector {
    using member_type = typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t** [6], DeviceType>::const_type element_freedom_table;
    typename Kokkos::View<double** [6], DeviceType>::const_type elements;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> vector;

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_nodes), [&](size_t i_node) {
            for (auto j = 0U; j < element_freedom_table.extent(2); ++j) {
                if constexpr (force_atomic) {
                    Kokkos::atomic_add(
                        &vector(element_freedom_table(i_elem, i_node, j), 0),
                        elements(i_elem, i_node, j)
                    );
                } else {
                    vector(element_freedom_table(i_elem, i_node, j), 0) +=
                        elements(i_elem, i_node, j);
                }
            }
        });
    }
};

}  // namespace openturbine
