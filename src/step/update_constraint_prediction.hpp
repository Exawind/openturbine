#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"
#include "constraints/update_lambda_prediction.hpp"
#include "solver/solver.hpp"

namespace openturbine {

template <typename DeviceType>
inline void UpdateConstraintPrediction(
    Solver<DeviceType>& solver, Constraints<DeviceType>& constraints
) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Prediction");
    auto range_policy =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, constraints.num_constraints);
    Kokkos::parallel_for(
        "UpdateLambdaPrediction", range_policy,
        UpdateLambdaPrediction<DeviceType>{
            solver.num_system_dofs, constraints.row_range, solver.x, constraints.lambda
        }
    );
}

}  // namespace openturbine
