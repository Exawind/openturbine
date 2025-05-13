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

template <typename DeviceType>
inline void UpdateStatePrediction(
    StepParameters& parameters, const Solver<DeviceType>& solver, State<DeviceType>& state
) {
    auto region = Kokkos::Profiling::ScopedRegion("Update State Prediction");
    auto range_policy =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, solver.num_system_nodes);
    if (parameters.is_dynamic_solve) {
        Kokkos::parallel_for(
            "UpdateDynamicPrediction", range_policy,
            UpdateDynamicPrediction<DeviceType>{
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
            "UpdateStaticPrediction", range_policy,
            UpdateStaticPrediction<DeviceType>{
                parameters.h,
                state.node_freedom_allocation_table,
                state.node_freedom_map_table,
                solver.x,
                state.q_delta,
            }
        );
    }

    Kokkos::parallel_for(
        "CalculateDisplacement", range_policy,
        CalculateDisplacement<DeviceType>{
            parameters.h,
            state.q_delta,
            state.q_prev,
            state.q,
        }
    );
}

}  // namespace openturbine
