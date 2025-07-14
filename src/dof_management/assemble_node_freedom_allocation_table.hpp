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
    void operator()(size_t element) const {
        const auto num_nodes = num_nodes_per_element(element);
        for (auto node = 0U; node < num_nodes; ++node) {
            const auto node_index = node_state_indices(element, node);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), element_freedom_signature(element, node)
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
    void operator()(size_t element) const {
        // Masses always have one node per element
        const auto node_index = node_state_indices(element);
        Kokkos::atomic_or(
            &node_freedom_allocation_table(node_index), element_freedom_signature(element)
        );
    }
};

template <typename DeviceType>
struct AssembleNodeFreedomMapTable_Springs {
    typename Kokkos::View<size_t* [2], DeviceType>::const_type node_state_indices;
    typename Kokkos::View<FreedomSignature* [2], DeviceType>::const_type element_freedom_signature;
    Kokkos::View<FreedomSignature*, DeviceType> node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t element) const {
        // Springs always have two nodes per element
        for (auto node = 0U; node < 2U; ++node) {
            const auto node_index = node_state_indices(element, node);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), element_freedom_signature(element, node)
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
    void operator()(size_t node) const {
        {
            const auto node_index = target_node_index(node);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), target_node_freedom_signature(node)
            );
        }

        if (GetNumberOfNodes(type(node)) == 2U) {
            const auto node_index = base_node_index(node);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), base_node_freedom_signature(node)
            );
        }
    }
};

template <typename DeviceType>
inline void assemble_node_freedom_allocation_table(
    State<DeviceType>& state, const Elements<DeviceType>& elements,
    const Constraints<DeviceType>& constraints
) {
    using RangePolicy = Kokkos::RangePolicy<typename DeviceType::execution_space>;
    Kokkos::deep_copy(state.node_freedom_allocation_table, FreedomSignature::NoComponents);

    auto beams_range = RangePolicy(0, elements.beams.num_elems);

    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Beams", beams_range,
        AssembleNodeFreedomMapTable_Beams<DeviceType>{
            elements.beams.num_nodes_per_element, elements.beams.node_state_indices,
            elements.beams.element_freedom_signature, state.node_freedom_allocation_table
        }
    );
    auto masses_range = RangePolicy(0, elements.masses.num_elems);
    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Masses", masses_range,
        AssembleNodeFreedomMapTable_Masses<DeviceType>{
            elements.masses.state_indices, elements.masses.element_freedom_signature,
            state.node_freedom_allocation_table
        }
    );
    auto springs_range = RangePolicy(0, elements.springs.num_elems);
    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Springs", springs_range,
        AssembleNodeFreedomMapTable_Springs<DeviceType>{
            elements.springs.node_state_indices, elements.springs.element_freedom_signature,
            state.node_freedom_allocation_table
        }
    );
    auto constraints_range = RangePolicy(0, constraints.num_constraints);
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
    auto system_range = RangePolicy(0, state.num_system_nodes);
    Kokkos::parallel_for(
        "ComputeActiveDofs", system_range,
        KOKKOS_LAMBDA(size_t node) {
            active_dofs(node) = count_active_dofs(node_freedom_allocation_table(node));
        }
    );
}

}  // namespace openturbine
