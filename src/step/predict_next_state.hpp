#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "state/calculate_displacement.hpp"
#include "state/calculate_next_state.hpp"
#include "state/state.hpp"
#include "step_parameters.hpp"

namespace openturbine {

template <typename DeviceType>
inline void PredictNextState(StepParameters& parameters, State<DeviceType>& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Predict Next State");

    using RangePolicy = Kokkos::RangePolicy<typename DeviceType::execution_space>;

    Kokkos::deep_copy(state.q_prev, state.q);

    Kokkos::parallel_for(
        "CalculateNextState", RangePolicy(0, state.v.extent(0)),
        CalculateNextState<DeviceType>{
            parameters.h,
            parameters.alpha_f,
            parameters.alpha_m,
            parameters.beta,
            parameters.gamma,
            state.q_delta,
            state.v,
            state.vd,
            state.a,
        }
    );

    Kokkos::parallel_for(
        "CalculateDisplacement", RangePolicy(0, state.q.extent(0)),
        CalculateDisplacement<DeviceType>{
            parameters.h,
            state.q_delta,
            state.q_prev,
            state.q,
        }
    );

    state.time_step++;
}

}  // namespace openturbine
