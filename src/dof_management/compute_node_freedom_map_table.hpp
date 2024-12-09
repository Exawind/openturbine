#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"

#include "src/beams/beams.hpp"
#include "src/state/state.hpp"

namespace openturbine {

struct ComputeNodeFreedomMapTable {
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*> node_freedom_map_table;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update, bool is_final) const {
        const auto num_dof = count_active_dofs(node_freedom_allocation_table(i));
        update += num_dof;
        if (is_final) {
            node_freedom_map_table(i+1) = update;
        }

    }
};

inline void compute_node_freedom_map_table(State& state) {
    Kokkos::deep_copy(state.node_freedom_map_table, 0UL);
    auto result = 0UL;
    Kokkos::parallel_scan("Compute Node Freedom Map Table", state.num_system_nodes, ComputeNodeFreedomMapTable{state.node_freedom_allocation_table, state.node_freedom_map_table}, result);
}

}  // namespace openturbine
