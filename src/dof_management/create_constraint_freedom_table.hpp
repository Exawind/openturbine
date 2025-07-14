#pragma once

#include <Kokkos_Core.hpp>

#include "constraints/constraints.hpp"
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
    void operator()(size_t constraint) const {
        {
            const auto node_index = target_node_index(constraint);
            const auto num_active_dofs = active_dofs(node_index);
            for (auto component = 0U; component < num_active_dofs; ++component) {
                target_node_freedom_table(constraint, component) =
                    node_freedom_map_table(node_index) + component;
            }
        }
        if (GetNumberOfNodes(type(constraint)) == 2U) {
            const auto node_index = base_node_index(constraint);
            const auto num_active_dofs = active_dofs(node_index);
            for (auto component = 0U; component < num_active_dofs; ++component) {
                base_node_freedom_table(constraint, component) =
                    node_freedom_map_table(node_index) + component;
            }
        } else {
            for (auto component = 0U; component < 6U; ++component) {
                base_node_freedom_table(constraint, component) = 0UL;
            }
        }
    }
};

template <typename DeviceType>
inline void create_constraint_freedom_table(
    Constraints<DeviceType>& constraints, const State<DeviceType>& state
) {
    using RangePolicy = Kokkos::RangePolicy<typename DeviceType::execution_space>;
    auto constraints_range = RangePolicy(0, constraints.num_constraints);

    Kokkos::parallel_for(
        "Create Constraint Node Freedom Table", constraints_range,
        CreateConstraintFreedomTable<DeviceType>{
            constraints.type, constraints.target_node_index, constraints.base_node_index,
            state.active_dofs, state.node_freedom_map_table, constraints.target_node_freedom_table,
            constraints.base_node_freedom_table
        }
    );
}

}  // namespace openturbine
