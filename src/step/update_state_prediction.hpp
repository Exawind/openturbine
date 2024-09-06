#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "step_parameters.hpp"

#include "src/solver/solver.hpp"
#include "src/state/calculate_displacement.hpp"
#include "src/state/state.hpp"
#include "src/state/update_dynamic_prediction.hpp"
#include "src/state/update_static_prediction.hpp"

namespace openturbine {

inline void UpdateStatePrediction(StepParameters& parameters, const Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Update State Prediction");
    const auto x_system =
        Kokkos::subview(solver.x, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs));
    if (parameters.is_dynamic_solve) {
        Kokkos::parallel_for(
            "UpdateDynamicPrediction", solver.num_system_nodes,
            UpdateDynamicPrediction{
                parameters.h,
                parameters.beta_prime,
                parameters.gamma_prime,
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
                parameters.h,
                parameters.beta_prime,
                parameters.gamma_prime,
                x_system,
                state.q_delta,
            }
        );
    }

    Kokkos::parallel_for(
        "CalculateDisplacement", solver.num_system_nodes,
        CalculateDisplacement{
            parameters.h,
            state.q_delta,
            state.q_prev,
            state.q,
        }
    );
}

}  // namespace openturbine
