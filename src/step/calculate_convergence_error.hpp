#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/solver/calculate_error_sum_squares.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/step/step_parameters.hpp"

namespace openturbine {

inline double CalculateConvergenceError(StepParameters& parameters, Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Calculate Convergence Error");
    const double atol = parameters.absolute_convergence_tol;
    const double rtol = parameters.relative_convergence_tol;
    double sum_error_squared = 0.;
    Kokkos::parallel_reduce(
        solver.num_system_dofs,
        CalculateErrorSumSquares{
            atol,
            rtol,
            state.q_delta,
            solver.x,
        },
        sum_error_squared
    );
    return std::sqrt(sum_error_squared / static_cast<double>(solver.num_system_dofs));
}

}  // namespace openturbine
