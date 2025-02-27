#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/calculate_constraint_residual_gradient.hpp"
#include "constraints/constraints.hpp"
#include "state/state.hpp"

namespace openturbine {

inline void UpdateConstraintVariables(State& state, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Variables");

    if (constraints.num_constraints == 0) {
        return;
    }

    constraints.UpdateViews();

    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", constraints.num_constraints,
        CalculateConstraintResidualGradient{
            constraints.type, constraints.target_node_col_range, constraints.base_node_index,
            constraints.target_node_index, constraints.X0, constraints.axes, constraints.input, constraints.lambda, state.tangent,
            state.q, constraints.residual_terms, constraints.base_lambda_residual_terms, constraints.target_lambda_residual_terms, constraints.system_residual_terms, constraints.base_gradient_terms,
            constraints.target_gradient_terms, constraints.base_gradient_transpose_terms, constraints.target_gradient_transpose_terms
        }
    );
}

}  // namespace openturbine
