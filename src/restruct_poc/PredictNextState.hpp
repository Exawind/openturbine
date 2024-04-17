#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "CalculateDisplacement.hpp"
#include "CalculateNextState.hpp"
#include "Solver.hpp"

namespace openturbine {

inline void PredictNextState(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Predict Next State");
    Kokkos::deep_copy(solver.state.lambda, 0.);
    Kokkos::deep_copy(solver.state.q_prev, solver.state.q);

    Kokkos::parallel_for(
        "CalculateNextState", solver.num_system_nodes,
        CalculateNextState{
            solver.h,
            solver.alpha_f,
            solver.alpha_m,
            solver.beta,
            solver.gamma,
            solver.state.q_delta,
            solver.state.v,
            solver.state.vd,
            solver.state.a,
        }
    );

    Kokkos::parallel_for(
        "CalculateDisplacement", solver.state.q.extent(0),
        CalculateDisplacement{
            solver.h,
            solver.state.q_delta,
            solver.state.q_prev,
            solver.state.q,
        }
    );
}

}  // namespace openturbine
