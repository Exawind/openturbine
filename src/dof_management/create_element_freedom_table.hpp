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
    void operator()(size_t i_elem) const {
        for (auto j_node = 0U; j_node < num_nodes_per_element(i_elem); ++j_node) {
            const auto node_index = node_state_indices(i_elem, j_node);
            for (auto k_dof = 0U; k_dof < 6U; ++k_dof) {
                element_freedom_table(i_elem, j_node, k_dof) =
                    node_freedom_map_table(node_index) + k_dof;
            }
        }
    }
};

struct CreateElementFreedomTable_Masses {
    Kokkos::View<size_t*>::const_type node_state_indices;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t**> element_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Masses always have one node per element
        const auto node_index = node_state_indices(i_elem);
        for (auto k_dof = 0U; k_dof < 6U; ++k_dof) {
            element_freedom_table(i_elem, k_dof) = node_freedom_map_table(node_index) + k_dof;
        }
    }
};

struct CreateElementFreedomTable_Springs {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t* [2]>::const_type node_state_indices;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t* [2][3]> element_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Springs always have two nodes per element
        for (auto j_node = 0U; j_node < 2U; ++j_node) {
            const auto node_index = node_state_indices(i_elem, j_node);
            // Springs only have translational DOFs
            for (auto k_dof = 0U; k_dof < 3U; ++k_dof) {
                element_freedom_table(i_elem, j_node, k_dof) =
                    node_freedom_map_table(node_index) + k_dof;
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
