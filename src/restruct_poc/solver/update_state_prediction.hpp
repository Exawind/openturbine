#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_displacement.hpp"
#include "solver.hpp"
#include "update_dynamic_prediction.hpp"
#include "update_lambda_prediction.hpp"
#include "update_static_prediction.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

inline void UpdateStatePrediction(
    const Solver& solver, const View_N& x_system, const View_N& x_lambda
) {
    auto region = Kokkos::Profiling::ScopedRegion("Update State Prediction");
    if (solver.is_dynamic_solve) {
        Kokkos::parallel_for(
            "UpdateDynamicPrediction", solver.num_system_nodes,
            UpdateDynamicPrediction{
                solver.h,
                solver.beta_prime,
                solver.gamma_prime,
                x_system,
                solver.state.q_delta,
                solver.state.v,
                solver.state.vd,
            }
        );
    } else {
        Kokkos::parallel_for(
            "UpdateStaticPrediction", solver.num_system_nodes,
            UpdateStaticPrediction{
                solver.h,
                solver.beta_prime,
                solver.gamma_prime,
                x_system,
                solver.state.q_delta,
            }
        );
    }

    Kokkos::parallel_for(
        "CalculateDisplacement", solver.num_system_nodes,
        CalculateDisplacement{
            solver.h,
            solver.state.q_delta,
            solver.state.q_prev,
            solver.state.q,
        }
    );

    if (solver.constraints.num > 0) {
        Kokkos::parallel_for(
            "UpdateLambdaPrediction", solver.constraints.num_dofs,
            UpdateLambdaPrediction{
                x_lambda,
                solver.state.lambda,
            }
        );
    }
}

}  // namespace openturbine
