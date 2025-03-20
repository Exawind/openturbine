#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver/solver.hpp"
#include "state/calculate_displacement.hpp"
#include "state/state.hpp"
#include "state/update_dynamic_prediction.hpp"
#include "state/update_static_prediction.hpp"
#include "step_parameters.hpp"

namespace openturbine {

inline void UpdateStatePrediction(StepParameters& parameters, const Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Update State Prediction");
    if (parameters.is_dynamic_solve) {
        Kokkos::parallel_for(
            "UpdateDynamicPrediction", solver.num_system_nodes,
            UpdateDynamicPrediction{
                parameters.h,
                parameters.beta_prime,
                parameters.gamma_prime,
                state.node_freedom_allocation_table,
                state.node_freedom_map_table,
                solver.x,
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
                state.node_freedom_allocation_table,
                state.node_freedom_map_table,
                solver.x,
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
