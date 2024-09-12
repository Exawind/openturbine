#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "step_parameters.hpp"

#include "src/state/state.hpp"
#include "src/system/calculate_tangent_operator.hpp"

namespace openturbine {

inline void UpdateTangentOperator(StepParameters& parameters, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Tangent Operator");
    Kokkos::parallel_for(
        "CalculateTangentOperator", state.num_system_nodes,
        CalculateTangentOperator{
            parameters.h,
            state.q_delta,
            state.tangent,
        }
    );
}

}  // namespace openturbine
