#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_constraint_residual_gradient.hpp"
#include "constraints.hpp"
#include "state.hpp"

namespace openturbine {

inline void UpdateConstraintVariables(State& state, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Variables");

    if (constraints.num == 0) {
        return;
    }

    constraints.UpdateViews();

    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", constraints.num,
        CalculateConstraintResidualGradient{
            constraints.type, constraints.node_index, constraints.row_range, constraints.node_col_range, constraints.X0, constraints.axis, constraints.control, constraints.u,
            state.q, constraints.Phi, constraints.gradient_terms}
    );
}

}  // namespace openturbine
