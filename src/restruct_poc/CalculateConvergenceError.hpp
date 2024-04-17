#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"
#include "CalculateErrorSumSquares.hpp"

namespace openturbine {

inline double CalculateConvergenceError(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Calculate Convergence Error");
    const double atol = 1e-7;
    const double rtol = 1e-5;
    double sum_error_squared = 0.;
    Kokkos::parallel_reduce(
        solver.num_system_dofs,
        CalculateErrorSumSquares{
            atol,
            rtol,
            solver.state.q_delta,
            solver.x,
        },
        sum_error_squared
    );
    return std::sqrt(sum_error_squared / solver.num_system_dofs);
}

}
