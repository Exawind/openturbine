#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct ContributeBeamsToVector {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t** [6]>::const_type element_freedom_table;
    Kokkos::View<double** [6]>::const_type elements;
    Kokkos::View<double*> vector;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        constexpr auto force_atomic =
            !std::is_same_v<Kokkos::TeamPolicy<>::member_type::execution_space, Kokkos::Serial>;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_nodes), [&](size_t i_node) {
            for (auto j = 0U; j < element_freedom_table.extent(2); ++j) {
                if constexpr (force_atomic) {
                    Kokkos::atomic_add(
                        &vector(element_freedom_table(i_elem, i_node, j)),
                        elements(i_elem, i_node, j)
                    );
                } else {
                    vector(element_freedom_table(i_elem, i_node, j)) += elements(i_elem, i_node, j);
                }
            }
        });
    }
};

}  // namespace openturbine
