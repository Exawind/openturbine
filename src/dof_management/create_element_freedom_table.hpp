#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/elements/elements.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void create_element_freedom_table(Elements& elements, const State& state) {
    // Beams data
    auto has_beams = elements.beams != nullptr;
    const auto beams_num_elems = has_beams ? elements.beams->num_elems : 0U;
    const auto beams_node_state_indices =
        has_beams ? elements.beams->node_state_indices
                  : Kokkos::View<size_t**>("beams_node_state_indices", 0);
    const auto beams_num_nodes_per_element =
        has_beams ? elements.beams->num_nodes_per_element
                  : Kokkos::View<size_t*>("beams_num_nodes_per_element", 0);
    auto beams_element_freedom_table =
        has_beams ? elements.beams->element_freedom_table
                  : Kokkos::View<size_t** [6]>("beams_element_freedom_table", 0);

    // Beams element freedom table
    Kokkos::parallel_for(
        "Create Beams Element Freedom Table", 1,
        KOKKOS_LAMBDA(size_t) {
            for (auto i = 0U; i < beams_num_elems; ++i) {
                const auto num_nodes = beams_num_nodes_per_element(i);
                for (auto j = 0U; j < num_nodes; ++j) {
                    const auto node_index = beams_node_state_indices(i, j);
                    for (auto k = 0U; k < 6U; ++k) {
                        beams_element_freedom_table(i, j, k) =
                            state.node_freedom_map_table(node_index) + k;
                    }
                }
            }
        }
    );

    // Masses data
    auto has_masses = elements.masses != nullptr;
    const auto masses_num_elems = has_masses ? elements.masses->num_elems : 0U;
    const auto masses_node_state_indices =
        has_masses ? elements.masses->state_indices
                   : Kokkos::View<size_t* [1]>("masses_node_state_indices", 0);
    auto masses_element_freedom_table =
        has_masses ? elements.masses->element_freedom_table
                   : Kokkos::View<size_t* [1][6]>("masses_element_freedom_table", 0);

    // Masses element freedom table
    Kokkos::parallel_for(
        "Create Masses Element Freedom Table", 1,
        KOKKOS_LAMBDA(size_t) {
            for (auto i = 0U; i < masses_num_elems; ++i) {
                const auto num_nodes = 1U;
                for (auto j = 0U; j < num_nodes; ++j) {
                    const auto node_index = masses_node_state_indices(i, j);
                    for (auto k = 0U; k < 6U; ++k) {
                        masses_element_freedom_table(i, j, k) =
                            state.node_freedom_map_table(node_index) + k;
                    }
                }
            }
        }
    );
}

}  // namespace openturbine
