#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_displacement.hpp"
#include "calculate_next_state.hpp"
#include "solver.hpp"
#include "state.hpp"

namespace openturbine {

inline void PredictNextState(Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Predict Next State");
    Kokkos::deep_copy(state.lambda, 0.);
    Kokkos::deep_copy(state.q_prev, state.q);

    Kokkos::parallel_for(
        "CalculateNextState", solver.num_system_nodes,
        CalculateNextState{
            solver.h,
            solver.alpha_f,
            solver.alpha_m,
            solver.beta,
            solver.gamma,
            state.q_delta,
            state.v,
            state.vd,
            state.a,
        }
    );

    Kokkos::parallel_for(
        "CalculateDisplacement", state.q.extent(0),
        CalculateDisplacement{
            solver.h,
            state.q_delta,
            state.q_prev,
            state.q,
        }
    );
}

}  // namespace openturbine
