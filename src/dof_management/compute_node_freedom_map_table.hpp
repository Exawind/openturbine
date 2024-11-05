#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/beams/beams.hpp"
#include "src/state/state.hpp"

namespace openturbine {

inline void compute_node_freedom_map_table(State& state) {
    Kokkos::parallel_for("Compute Node Freedom Map Table", 1, KOKKOS_LAMBDA(size_t) {
        state.node_freedom_map_table(0) = 0U;
        for(auto i = 1U; i < state.num_system_nodes; ++i) {
            const auto num_dof = count_active_dofs(state.node_freedom_allocation_table(i-1));
            state.node_freedom_map_table(i) = state.node_freedom_map_table(i-1) + num_dof;
        }
    });
}

}
