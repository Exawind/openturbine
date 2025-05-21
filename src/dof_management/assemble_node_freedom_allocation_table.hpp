#pragma once

#include <Kokkos_Core.hpp>

#include "constraints/constraints.hpp"
#include "elements/elements.hpp"
#include "freedom_signature.hpp"
#include "state/state.hpp"

namespace openturbine {

template <typename DeviceType>
struct AssembleNodeFreedomMapTable_Beams {
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t**, DeviceType>::const_type node_state_indices;
    typename Kokkos::View<FreedomSignature**, DeviceType>::const_type element_freedom_signature;
    Kokkos::View<FreedomSignature*, DeviceType> node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        const auto num_nodes = num_nodes_per_element(i_elem);
        for (auto j_node = 0U; j_node < num_nodes; ++j_node) {
            const auto node_index = node_state_indices(i_elem, j_node);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), element_freedom_signature(i_elem, j_node)
            );
        }
    }
};

template <typename DeviceType>
struct AssembleNodeFreedomMapTable_Masses {
    typename Kokkos::View<size_t*, DeviceType>::const_type node_state_indices;
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type element_freedom_signature;
    Kokkos::View<FreedomSignature*, DeviceType> node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Masses always have one node per element
        const auto node_index = node_state_indices(i_elem);
        Kokkos::atomic_or(
            &node_freedom_allocation_table(node_index), element_freedom_signature(i_elem)
        );
    }
};

template <typename DeviceType>
struct AssembleNodeFreedomMapTable_Springs {
    typename Kokkos::View<size_t* [2], DeviceType>::const_type node_state_indices;
    typename Kokkos::View<FreedomSignature* [2], DeviceType>::const_type element_freedom_signature;
    Kokkos::View<FreedomSignature*, DeviceType> node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Springs always have two nodes per element
        for (auto j_node = 0U; j_node < 2U; ++j_node) {
            const auto node_index = node_state_indices(i_elem, j_node);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), element_freedom_signature(i_elem, j_node)
            );
        }
    }
};

template <typename DeviceType>
struct AssembleNodeFreedomMapTable_Constraints {
    typename Kokkos::View<ConstraintType*, DeviceType>::const_type type;
    typename Kokkos::View<size_t*, DeviceType>::const_type target_node_index;
    typename Kokkos::View<size_t*, DeviceType>::const_type base_node_index;
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type target_node_freedom_signature;
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type base_node_freedom_signature;
    Kokkos::View<FreedomSignature*, DeviceType> node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        {
            const auto node_index = target_node_index(i);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), target_node_freedom_signature(i)
            );
        }

        if (GetNumberOfNodes(type(i)) == 2U) {
            const auto node_index = base_node_index(i);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), base_node_freedom_signature(i)
            );
        }
    }
};

template <typename DeviceType>
inline void assemble_node_freedom_allocation_table(
    State<DeviceType>& state, const Elements<DeviceType>& elements,
    const Constraints<DeviceType>& constraints
) {
    Kokkos::deep_copy(state.node_freedom_allocation_table, FreedomSignature::NoComponents);

    auto beams_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, elements.beams.num_elems);

    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Beams", beams_range,
        AssembleNodeFreedomMapTable_Beams<DeviceType>{
            elements.beams.num_nodes_per_element, elements.beams.node_state_indices,
            elements.beams.element_freedom_signature, state.node_freedom_allocation_table
        }
    );
    auto masses_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, elements.masses.num_elems);
    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Masses", masses_range,
        AssembleNodeFreedomMapTable_Masses<DeviceType>{
            elements.masses.state_indices, elements.masses.element_freedom_signature,
            state.node_freedom_allocation_table
        }
    );
    auto springs_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, elements.springs.num_elems);
    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Springs", springs_range,
        AssembleNodeFreedomMapTable_Springs<DeviceType>{
            elements.springs.node_state_indices, elements.springs.element_freedom_signature,
            state.node_freedom_allocation_table
        }
    );
    auto constraints_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, constraints.num_constraints);
    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Constraints", constraints_range,
        AssembleNodeFreedomMapTable_Constraints<DeviceType>{
            constraints.type, constraints.target_node_index, constraints.base_node_index,
            constraints.target_node_freedom_signature, constraints.base_node_freedom_signature,
            state.node_freedom_allocation_table
        }
    );

    const auto active_dofs = state.active_dofs;
    const auto node_freedom_allocation_table = state.node_freedom_allocation_table;
    auto system_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, state.num_system_nodes);
    Kokkos::parallel_for(
        "ComputeActiveDofs", system_range,
        KOKKOS_LAMBDA(size_t i) {
            active_dofs(i) = count_active_dofs(node_freedom_allocation_table(i));
        }
    );
}

}  // namespace openturbine
