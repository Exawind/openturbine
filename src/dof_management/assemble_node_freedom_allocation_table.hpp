#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/beams/beams.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void assemble_node_freedom_allocation_table(State& state, const Beams& beams) {
    Kokkos::parallel_for("Assemble Node Freedom Map Table", 1, KOKKOS_LAMBDA(size_t) {
        for(auto i = 0U; i < beams.num_elems; ++i) {
            const auto num_nodes = beams.num_nodes_per_element(i);
            for(auto j = 0U; j < num_nodes; ++j) {
                const auto node_index = beams.node_state_indices(i, j);
                const auto current_signature = state.node_freedom_allocation_table(node_index);
                const auto contributed_signature = beams.element_freedom_signature(i, j);
                state.node_freedom_allocation_table(node_index) = current_signature | contributed_signature;
            }
        }
    });
}

}
