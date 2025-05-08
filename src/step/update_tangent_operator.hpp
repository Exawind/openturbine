#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "state/state.hpp"
#include "step_parameters.hpp"
#include "system/calculate_tangent_operator.hpp"

namespace openturbine {

template <typename DeviceType>
inline void UpdateTangentOperator(StepParameters& parameters, State<DeviceType>& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Tangent Operator");
    Kokkos::parallel_for(
        "CalculateTangentOperator", Kokkos::RangePolicy<typename DeviceType::execution_space>(0, state.num_system_nodes),
        CalculateTangentOperator<DeviceType>{
            parameters.h,
            state.q_delta,
            state.tangent,
        }
    );
}

}  // namespace openturbine
