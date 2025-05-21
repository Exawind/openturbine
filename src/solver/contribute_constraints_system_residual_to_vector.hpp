#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct ContributeConstraintsSystemResidualToVector {
    Kokkos::View<size_t* [6]>::const_type target_node_freedom_table;
    Kokkos::View<size_t*>::const_type target_active_dofs;
    Kokkos::View<double* [6]>::const_type system_residual_terms;
    Kokkos::View<double* [1], Kokkos::LayoutLeft> residual;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
        const auto num_dofs = target_active_dofs(i_constraint);
        for (auto i = 0U; i < num_dofs; ++i) {
            residual(target_node_freedom_table(i_constraint, i), 0) +=
                system_residual_terms(i_constraint, i);
        }
    }
};

}  // namespace openturbine
