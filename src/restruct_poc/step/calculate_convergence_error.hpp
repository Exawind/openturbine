#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/restruct_poc/solver/calculate_error_sum_squares.hpp"
#include "src/restruct_poc/solver/solver.hpp"
#include "src/restruct_poc/state/state.hpp"

namespace openturbine {

inline double CalculateConvergenceError(Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Calculate Convergence Error");
    const double atol = 1e-7;
    const double rtol = 1e-5;
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
