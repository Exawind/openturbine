#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/elements/elements.hpp"
#include "src/state/state.hpp"

namespace openturbine {

struct CreateElementFreedomTable_Beams {
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

struct CreateElementFreedomTable_Masses {
    Kokkos::View<size_t*>::const_type node_state_indices;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t**> element_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        // Masses always have one node per element
        const auto node_index = node_state_indices(i);
        for (auto k = 0U; k < 6U; ++k) {
            element_freedom_table(i, k) = node_freedom_map_table(node_index) + k;
        }
    }
};

struct CreateElementFreedomTable_Springs {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t* [2]>::const_type node_state_indices;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t* [2][3]> element_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        // Springs always have two nodes per element
        for (auto j = 0U; j < 2U; ++j) {
            const auto node_index = node_state_indices(i, j);
            // Springs only have translational DOFs
            for (auto k = 0U; k < 3U; ++k) {
                element_freedom_table(i, j, k) = node_freedom_map_table(node_index) + k;
            }
        }
    }
};

inline void create_element_freedom_table(Elements& elements, const State& state) {
    Kokkos::parallel_for(
        "CreateElementFreedomTable_Beams", elements.beams.num_elems,
        CreateElementFreedomTable_Beams{
            elements.beams.num_nodes_per_element, elements.beams.node_state_indices,
            state.node_freedom_map_table, elements.beams.element_freedom_table
        }
    );

    Kokkos::parallel_for(
        "CreateElementFreedomTable_Masses", elements.masses.num_elems,
        CreateElementFreedomTable_Masses{
            elements.masses.state_indices, state.node_freedom_map_table,
            elements.masses.element_freedom_table
        }
    );

    Kokkos::parallel_for(
        "CreateElementFreedomTable_Springs", elements.springs.num_elems,
        CreateElementFreedomTable_Springs{
            elements.springs.num_nodes_per_element, elements.springs.node_state_indices,
            state.node_freedom_map_table, elements.springs.element_freedom_table
        }
    );
}

}  // namespace openturbine
