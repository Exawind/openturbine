#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/constraints/constraints.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void create_constraint_freedom_table(Constraints& constraints, const State& state) {
    const auto constraints_num = constraints.num;
    const auto constraints_type = constraints.type;
    const auto constraints_target_node_index = constraints.target_node_index;
    const auto constraints_base_node_index = constraints.base_node_index;
    const auto constraints_target_node_freedom_table = constraints.target_node_freedom_table;
    const auto constraints_base_node_freedom_table = constraints.base_node_freedom_table;

    const auto state_node_freedom_map_table = state.node_freedom_map_table;

    Kokkos::parallel_for(
        "Create Constraint Node Freedom Table", 1,
        KOKKOS_LAMBDA(size_t) {
            for (auto i = 0U; i < constraints_num; ++i) {
                {
                    const auto node_index = constraints_target_node_index(i);
                    for (auto k = 0U; k < 6U; ++k) {
                        constraints_target_node_freedom_table(i, k) =
                            state_node_freedom_map_table(node_index) + k;
                    }
                }
                if (GetNumberOfNodes(constraints_type(i)) == 2U) {
                    const auto node_index = constraints_base_node_index(i);
                    for (auto k = 0U; k < 6U; ++k) {
                        constraints_base_node_freedom_table(i, k) =
                            state_node_freedom_map_table(node_index) + k;
                    }
                }
            }
        }
    );
}

}  // namespace openturbine
