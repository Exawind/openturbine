#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/beams/beams.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void create_element_freedom_table(Beams& beams, const State& state) {
    Kokkos::parallel_for(
        "Create Element Freedom Table", 1,
        KOKKOS_LAMBDA(size_t) {
            for (auto i = 0U; i < beams.num_elems; ++i) {
                const auto num_nodes = beams.num_nodes_per_element(i);
                for (auto j = 0U; j < num_nodes; ++j) {
                    const auto node_index = beams.node_state_indices(i, j);
                    for (auto k = 0U; k < 7U; ++k) {
                        beams.element_freedom_table(i, j, k) =
                            state.node_freedom_map_table(node_index) + k;
                    }
                }
            }
        }
    );
}

}  // namespace openturbine
