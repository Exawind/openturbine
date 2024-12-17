#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/constraints/constraints.hpp"
#include "src/elements/elements.hpp"
#include "src/state/state.hpp"

namespace openturbine {

struct AssembleNodeFreedomMapTable_Beams {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<FreedomSignature**>::const_type element_freedom_signature;
    Kokkos::View<FreedomSignature*> node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto num_nodes = num_nodes_per_element(i);
        for (auto j = 0U; j < num_nodes; ++j) {
            const auto node_index = node_state_indices(i, j);
            Kokkos::atomic_or(
                &node_freedom_allocation_table(node_index), element_freedom_signature(i, j)
            );
        }
    }
};

struct AssembleNodeFreedomMapTable_Masses {
    Kokkos::View<size_t* [1]>::const_type node_state_indices;
    Kokkos::View<FreedomSignature* [1]>::const_type element_freedom_signature;
    Kokkos::View<FreedomSignature*> node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto node_index = node_state_indices(i, 0);
        Kokkos::atomic_or(
            &node_freedom_allocation_table(node_index), element_freedom_signature(i, 0)
        );
    }
};

struct AssembleNodeFreedomMapTable_Constraints {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<FreedomSignature*>::const_type target_node_freedom_signature;
    Kokkos::View<FreedomSignature*>::const_type base_node_freedom_signature;
    Kokkos::View<FreedomSignature*> node_freedom_allocation_table;

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

inline void assemble_node_freedom_allocation_table(
    State& state, const Elements& elements, const Constraints& constraints
) {
    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Beams", elements.beams.num_elems,
        AssembleNodeFreedomMapTable_Beams{
            elements.beams.num_nodes_per_element, elements.beams.node_state_indices,
            elements.beams.element_freedom_signature, state.node_freedom_allocation_table
        }
    );
    
    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Masses", elements.masses.num_elems,
        AssembleNodeFreedomMapTable_Masses{
            elements.masses.state_indices, elements.masses.element_freedom_signature,
            state.node_freedom_allocation_table
        }
    );

    Kokkos::parallel_for(
        "AssembleNodeFreedomMapTable_Constraints", constraints.num,
        AssembleNodeFreedomMapTable_Constraints{
            constraints.type, constraints.target_node_index, constraints.base_node_index,
            constraints.target_node_freedom_signature, constraints.base_node_freedom_signature,
            state.node_freedom_allocation_table
        }
    );
}

}  // namespace openturbine
