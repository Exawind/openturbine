#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/restruct_poc/constraints/calculate_constraint_residual_gradient.hpp"
#include "src/restruct_poc/constraints/constraints.hpp"
#include "src/restruct_poc/state/state.hpp"

namespace openturbine {

inline void UpdateConstraintVariables(State& state, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Variables");

    if (constraints.num == 0) {
        return;
    }

    constraints.UpdateViews();

    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", constraints.num,
        CalculateConstraintResidualGradient{constraints.type, constraints.base_node_index, constraints.target_node_index, constraints.X0, constraints.axes, constraints.control, constraints.u, state.q, constraints.residual_terms, constraints.base_gradient_terms, constraints.target_gradient_terms}
    );
}

}  // namespace openturbine
