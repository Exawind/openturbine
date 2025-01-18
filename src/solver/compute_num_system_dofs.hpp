#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

struct ComputeNumSystemDofsReducer {
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update) const {
        update += count_active_dofs(node_freedom_allocation_table(i));
    }
};

/// Computes the total number of active degrees of freedom in the system
[[nodiscard]] inline size_t ComputeNumSystemDofs(
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table
) {
    auto total_system_dofs = 0UL;
    Kokkos::parallel_reduce(
        "ComputeNumSystemDofs", node_freedom_allocation_table.extent(0),
        ComputeNumSystemDofsReducer{node_freedom_allocation_table}, total_system_dofs
    );
    return total_system_dofs;
}

}  // namespace openturbine
