#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "step_parameters.hpp"

#include "src/solver/solver.hpp"
#include "src/state/calculate_displacement.hpp"
#include "src/state/calculate_next_state.hpp"
#include "src/state/state.hpp"

namespace openturbine {

void PredictNextState(const StepParameters& parameters, const State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Predict Next State");
    Kokkos::deep_copy(state.q_prev, state.q);

    Kokkos::parallel_for(
        "CalculateNextState", state.v.extent(0),
        CalculateNextState{
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
        "CalculateDisplacement", state.q.extent(0),
        CalculateDisplacement{
            parameters.h,
            state.q_delta,
            state.q_prev,
            state.q,
        }
    );
}

}  // namespace openturbine
