#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"
#include "constraints/update_lambda_prediction.hpp"
#include "solver/solver.hpp"

namespace openturbine {

template <typename DeviceType>
inline void UpdateConstraintPrediction(Solver<DeviceType>& solver, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Prediction");
    Kokkos::parallel_for(
        "UpdateLambdaPrediction", constraints.num_constraints,
        UpdateLambdaPrediction{
            solver.num_system_dofs, constraints.row_range, solver.x, constraints.lambda
        }
    );
}

}  // namespace openturbine
