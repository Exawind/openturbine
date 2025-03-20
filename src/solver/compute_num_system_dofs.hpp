#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

struct ComputeNumSystemDofsReducer {
    Kokkos::View<size_t*>::const_type active_dofs;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update) const { update += active_dofs(i); }
};

/// Computes the total number of active degrees of freedom in the system
[[nodiscard]] inline size_t ComputeNumSystemDofs(const Kokkos::View<size_t*>::const_type& active_dofs
) {
    auto total_system_dofs = 0UL;
    Kokkos::parallel_reduce(
        "ComputeNumSystemDofs", active_dofs.extent(0), ComputeNumSystemDofsReducer{active_dofs},
        total_system_dofs
    );
    return total_system_dofs;
}

}  // namespace openturbine
