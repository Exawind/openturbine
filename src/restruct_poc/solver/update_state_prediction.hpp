#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_displacement.hpp"
#include "solver.hpp"
#include "state.hpp"
#include "update_dynamic_prediction.hpp"
#include "update_lambda_prediction.hpp"
#include "update_static_prediction.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

inline void UpdateStatePrediction(const Solver& solver, State& state, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Update State Prediction");
    const auto x_system = Kokkos::subview(solver.x, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs));
    const auto x_lambda = Kokkos::subview(solver.x, Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs));
    if (solver.is_dynamic_solve) {
        Kokkos::parallel_for(
            "UpdateDynamicPrediction", solver.num_system_nodes,
            UpdateDynamicPrediction{
                solver.h,
                solver.beta_prime,
                solver.gamma_prime,
                x_system,
                state.q_delta,
                state.v,
                state.vd,
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
                state.q_delta,
            }
        );
    }

    Kokkos::parallel_for(
        "CalculateDisplacement", solver.num_system_nodes,
        CalculateDisplacement{
            solver.h,
            state.q_delta,
            state.q_prev,
            state.q,
        }
    );

    if (constraints.num > 0) {
        Kokkos::parallel_for(
            "UpdateLambdaPrediction", constraints.num_dofs,
            UpdateLambdaPrediction{
                x_lambda,
                state.lambda,
            }
        );
    }
}

}  // namespace openturbine
