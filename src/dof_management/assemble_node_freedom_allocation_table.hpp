#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/beams/beams.hpp"
#include "src/constraints/constraints.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void assemble_node_freedom_allocation_table(
    State& state, const Beams& beams, const Constraints& constraints
) {
    const auto state_node_freedom_allocation_table = state.node_freedom_allocation_table;

    const auto beams_num_elems = beams.num_elems;
    const auto beams_num_nodes_per_element = beams.num_nodes_per_element;
    const auto beams_node_state_indices = beams.node_state_indices;
    const auto beams_element_freedom_signature = beams.element_freedom_signature;

    const auto constraints_num = constraints.num;
    const auto constraints_type = constraints.type;
    const auto constraints_target_node_index = constraints.target_node_index;
    const auto constraints_target_node_freedom_signature = constraints.target_node_freedom_signature;
    const auto constraints_base_node_index = constraints.base_node_index;
    const auto constraints_base_node_freedom_signature = constraints.base_node_freedom_signature;
    Kokkos::parallel_for(
        "Assemble Node Freedom Map Table", 1,
        KOKKOS_LAMBDA(size_t) {
            for (auto i = 0U; i < beams_num_elems; ++i) {
                const auto num_nodes = beams_num_nodes_per_element(i);
                for (auto j = 0U; j < num_nodes; ++j) {
                    const auto node_index = beams_node_state_indices(i, j);
                    const auto current_signature = state_node_freedom_allocation_table(node_index);
                    const auto contributed_signature = beams_element_freedom_signature(i, j);
                    state_node_freedom_allocation_table(node_index) =
                        current_signature | contributed_signature;
                }
            }

            for (auto i = 0U; i < constraints_num; ++i) {
                {
                    const auto node_index = constraints_target_node_index(i);
                    const auto current_signature = state_node_freedom_allocation_table(node_index);
                    const auto contributed_signature = constraints_target_node_freedom_signature(i);
                    state.node_freedom_allocation_table(node_index) =
                        current_signature | contributed_signature;
                }

                if (GetNumberOfNodes(constraints_type(i)) == 2U) {
                    const auto node_index = constraints_base_node_index(i);
                    const auto current_signature = state_node_freedom_allocation_table(node_index);
                    const auto contributed_signature = constraints_base_node_freedom_signature(i);
                    state_node_freedom_allocation_table(node_index) =
                        current_signature | contributed_signature;
                }
            }
        }
    );
}

}  // namespace openturbine
