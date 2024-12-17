#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/elements/elements.hpp"
#include "src/state/state.hpp"

namespace openturbine {

struct CreateElementFreedomTable {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t***> element_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        for (auto j = 0U; j < num_nodes_per_element(i); ++j) {
            const auto node_index = node_state_indices(i, j);
            for (auto k = 0U; k < 6U; ++k) {
                element_freedom_table(i, j, k) = node_freedom_map_table(node_index) + k;
            }
        }
    }
};

inline void create_element_freedom_table(Elements& elements, const State& state) {
    Kokkos::parallel_for(
        "Create Element Freedom Table", elements.beams.num_elems,
        CreateElementFreedomTable{
            elements.beams.num_nodes_per_element, elements.beams.node_state_indices,
            state.node_freedom_map_table, elements.beams.element_freedom_table
        }
    );
    
    Kokkos::parallel_for(
        "Create Element Freedom Table", elements.masses.num_elems,
        CreateElementFreedomTable{
            elements.masses.num_nodes_per_element, elements.masses.state_indices,
            state.node_freedom_map_table, elements.masses.element_freedom_table
        }
    );
}

}  // namespace openturbine
