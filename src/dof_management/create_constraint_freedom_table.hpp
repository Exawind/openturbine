#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "constraints/constraints.hpp"
#include "state/state.hpp"

namespace openturbine {

struct CreateConstraintFreedomTable {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t* [6]> target_node_freedom_table;
    Kokkos::View<size_t* [6]> base_node_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        {
            const auto node_index = target_node_index(i);
            const auto num_active_dofs =
                count_active_dofs(node_freedom_allocation_table(node_index));
            for (auto k = 0U; k < num_active_dofs; ++k) {
                target_node_freedom_table(i, k) = node_freedom_map_table(node_index) + k;
            }
        }
        if (GetNumberOfNodes(type(i)) == 2U) {
            const auto node_index = base_node_index(i);
            const auto num_active_dofs =
                count_active_dofs(node_freedom_allocation_table(node_index));
            for (auto k = 0U; k < num_active_dofs; ++k) {
                base_node_freedom_table(i, k) = node_freedom_map_table(node_index) + k;
            }
        }
    }
};

inline void create_constraint_freedom_table(Constraints& constraints, const State& state) {
    const auto constraints_type = constraints.type;
    const auto constraints_target_node_index = constraints.target_node_index;
    const auto constraints_base_node_index = constraints.base_node_index;
    const auto constraints_target_node_freedom_table = constraints.target_node_freedom_table;
    const auto constraints_base_node_freedom_table = constraints.base_node_freedom_table;

    const auto state_node_freedom_allocation_table = state.node_freedom_allocation_table;
    const auto state_node_freedom_map_table = state.node_freedom_map_table;

    Kokkos::parallel_for(
        "Create Constraint Node Freedom Table", constraints.num_constraints,
        CreateConstraintFreedomTable{
            constraints.type, constraints.target_node_index, constraints.base_node_index,
            state_node_freedom_allocation_table, state_node_freedom_map_table,
            constraints.target_node_freedom_table, constraints.base_node_freedom_table
        }
    );
}

}  // namespace openturbine
