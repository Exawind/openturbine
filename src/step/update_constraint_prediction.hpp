#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/constraints/constraints.hpp"
#include "src/constraints/update_lambda_prediction.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void UpdateConstraintPrediction(Solver& solver, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Prediction");
    const auto x_lambda =
        Kokkos::subview(solver.x, Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs));

    if (constraints.num > 0) {
        Kokkos::parallel_for(
            "UpdateLambdaPrediction", constraints.num_dofs,
            UpdateLambdaPrediction{
                x_lambda,
                constraints.lambda,
            }
        );
    }
}

}  // namespace openturbine