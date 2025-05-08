#pragma once

#include <Kokkos_Core.hpp>

#include "constraints/constraints.hpp"
#include "freedom_signature.hpp"
#include "state/state.hpp"

namespace openturbine {

template <typename DeviceType>
struct CreateConstraintFreedomTable {
    typename Kokkos::View<ConstraintType*, DeviceType>::const_type type;
    typename Kokkos::View<size_t*, DeviceType>::const_type target_node_index;
    typename Kokkos::View<size_t*, DeviceType>::const_type base_node_index;
    typename Kokkos::View<size_t*, DeviceType>::const_type active_dofs;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_freedom_map_table;
    Kokkos::View<size_t* [6], DeviceType> target_node_freedom_table;
    Kokkos::View<size_t* [6], DeviceType> base_node_freedom_table;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        {
            const auto node_index = target_node_index(i);
            const auto num_active_dofs = active_dofs(node_index);
            for (auto k = 0U; k < num_active_dofs; ++k) {
                target_node_freedom_table(i, k) = node_freedom_map_table(node_index) + k;
            }
        }
        if (GetNumberOfNodes(type(i)) == 2U) {
            const auto node_index = base_node_index(i);
            const auto num_active_dofs = active_dofs(node_index);
            for (auto k = 0U; k < num_active_dofs; ++k) {
                base_node_freedom_table(i, k) = node_freedom_map_table(node_index) + k;
            }
        } else {
            for (auto k = 0U; k < 6U; ++k) {
                base_node_freedom_table(i, k) = 0UL;
            }
        }
    }
};

template <typename DeviceType>
inline void create_constraint_freedom_table(
    Constraints<DeviceType>& constraints, const State<DeviceType>& state
) {
    Kokkos::parallel_for(
        "Create Constraint Node Freedom Table", Kokkos::RangePolicy<typename DeviceType::execution_space>(0, constraints.num_constraints),
        CreateConstraintFreedomTable<DeviceType>{
            constraints.type, constraints.target_node_index, constraints.base_node_index,
            state.active_dofs, state.node_freedom_map_table, constraints.target_node_freedom_table,
            constraints.base_node_freedom_table
        }
    );
}

}  // namespace openturbine
