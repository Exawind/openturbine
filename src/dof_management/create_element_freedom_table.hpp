#pragma once

#include <Kokkos_Core.hpp>

#include "elements/elements.hpp"
#include "freedom_signature.hpp"
#include "state/state.hpp"

namespace openturbine {

template <typename DeviceType>
struct CreateElementFreedomTable_Beams {
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t**, DeviceType>::const_type node_state_indices;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_freedom_map_table;
    Kokkos::View<size_t***, DeviceType> element_freedom_table;

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

template <typename DeviceType>
struct CreateElementFreedomTable_Masses {
    typename Kokkos::View<size_t*, DeviceType>::const_type node_state_indices;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_freedom_map_table;
    Kokkos::View<size_t**, DeviceType> element_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Masses always have one node per element
        const auto node_index = node_state_indices(i_elem);
        for (auto k_dof = 0U; k_dof < 6U; ++k_dof) {
            element_freedom_table(i_elem, k_dof) = node_freedom_map_table(node_index) + k_dof;
        }
    }
};

template <typename DeviceType>
struct CreateElementFreedomTable_Springs {
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t* [2], DeviceType>::const_type node_state_indices;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_freedom_map_table;
    Kokkos::View<size_t* [2][3], DeviceType> element_freedom_table;

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

template <typename DeviceType>
inline void create_element_freedom_table(
    Elements<DeviceType>& elements, const State<DeviceType>& state
) {
    auto beams_range = Kokkos::RangePolicy<typename DeviceType::execution_space>(0, elements.beams.num_elems);
    Kokkos::parallel_for(
        "CreateElementFreedomTable_Beams", beams_range,
        CreateElementFreedomTable_Beams<DeviceType>{
            elements.beams.num_nodes_per_element, elements.beams.node_state_indices,
            state.node_freedom_map_table, elements.beams.element_freedom_table
        }
    );
    auto masses_range = Kokkos::RangePolicy<typename DeviceType::execution_space>(0, elements.masses.num_elems);
    Kokkos::parallel_for(
        "CreateElementFreedomTable_Masses", masses_range,
        CreateElementFreedomTable_Masses<DeviceType>{
            elements.masses.state_indices, state.node_freedom_map_table,
            elements.masses.element_freedom_table
        }
    );
    auto springs_range = Kokkos::RangePolicy<typename DeviceType::execution_space>(0, elements.springs.num_elems);
    Kokkos::parallel_for(
        "CreateElementFreedomTable_Springs", springs_range,
        CreateElementFreedomTable_Springs<DeviceType>{
            elements.springs.num_nodes_per_element, elements.springs.node_state_indices,
            state.node_freedom_map_table, elements.springs.element_freedom_table
        }
    );
}

}  // namespace openturbine
