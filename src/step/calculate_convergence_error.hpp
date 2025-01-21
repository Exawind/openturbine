#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"
#include "solver/calculate_error_sum_squares.hpp"
#include "solver/solver.hpp"
#include "state/state.hpp"
#include "step/step_parameters.hpp"

namespace openturbine {

/// @brief Calculation based on Table 1 of DOI: 10.1115/1.4033441
inline double CalculateConvergenceError(
    const StepParameters& parameters, const Solver& solver, const State& state,
    const Constraints& constraints
) {
    auto region = Kokkos::Profiling::ScopedRegion("Calculate Convergence Error");
    double sum_error_squared = 0.;
    Kokkos::parallel_reduce(
        solver.num_system_nodes,
        CalculateSystemErrorSumSquares{
            parameters.absolute_convergence_tol,
            parameters.relative_convergence_tol,
            parameters.h,
            state.node_freedom_allocation_table,
            state.node_freedom_map_table,
            state.q_delta,
            solver.x,
        },
        sum_error_squared
    );
    Kokkos::parallel_reduce(
        constraints.num_dofs,
        CalculateConstraintsErrorSumSquares{
            parameters.absolute_convergence_tol,
            parameters.relative_convergence_tol,
            solver.num_system_dofs,
            constraints.lambda,
            solver.x,
        },
        sum_error_squared
    );

    return std::sqrt(sum_error_squared / static_cast<double>(solver.num_dofs));
}

}  // namespace openturbine
