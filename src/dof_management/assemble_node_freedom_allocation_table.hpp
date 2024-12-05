#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/constraints/constraints.hpp"
#include "src/elements/elements.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void assemble_node_freedom_allocation_table(
    State& state, const Elements& elements, const Constraints& constraints
) {
    const auto state_node_freedom_allocation_table = state.node_freedom_allocation_table;

    // Beams data
    auto has_beams = elements.beams != nullptr;
    const auto beams_num_elems = has_beams ? elements.beams->num_elems : 0U;
    const auto beams_node_state_indices =
        has_beams ? elements.beams->node_state_indices
                  : Kokkos::View<size_t**>("beams_node_state_indices", 0);
    const auto beams_num_nodes_per_element =
        has_beams ? elements.beams->num_nodes_per_element
                  : Kokkos::View<size_t*>("beams_num_nodes_per_element", 0);
    const auto beams_element_freedom_signature =
        has_beams ? elements.beams->element_freedom_signature
                  : Kokkos::View<FreedomSignature**>("beams_element_freedom_signature", 0);

    // Masses data
    auto has_masses = elements.masses != nullptr;
    const auto masses_num_elems = has_masses ? elements.masses->num_elems : 0U;
    const auto masses_node_state_indices =
        has_masses ? elements.masses->state_indices
                   : Kokkos::View<size_t*>("masses_node_state_indices", 0);
    const auto masses_element_freedom_signature =
        has_masses ? elements.masses->element_freedom_signature
                   : Kokkos::View<FreedomSignature*>("masses_element_freedom_signature", 0);

    // Constraints data
    const auto constraints_num = constraints.num;
    const auto constraints_type = constraints.type;
    const auto constraints_target_node_index = constraints.target_node_index;
    const auto constraints_target_node_freedom_signature = constraints.target_node_freedom_signature;
    const auto constraints_base_node_index = constraints.base_node_index;
    const auto constraints_base_node_freedom_signature = constraints.base_node_freedom_signature;

    Kokkos::parallel_for(
        "Assemble Node Freedom Map Table", 1,
        KOKKOS_LAMBDA(size_t) {
            // Assemble beam element contributions
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

            // Assemble mass element contributions
            for (auto i = 0U; i < masses_num_elems; ++i) {
                const auto node_index = masses_node_state_indices(i);
                const auto current_signature = state_node_freedom_allocation_table(node_index);
                const auto contributed_signature = masses_element_freedom_signature(i);
                state_node_freedom_allocation_table(node_index) =
                    current_signature | contributed_signature;
            }

            // Assemble constraint contributions
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
