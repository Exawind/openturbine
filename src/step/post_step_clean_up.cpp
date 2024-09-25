#include "step_parameters.hpp"

#include "src/beams/beams.hpp"
#include "src/constraints/calculate_constraint_output.hpp"
#include "src/constraints/constraints.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/state/update_algorithmic_acceleration.hpp"
#include "src/state/update_global_position.hpp"

namespace openturbine {

void PostStepCleanUp(const StepParameters& parameters, const Solver& solver, const State& state, const Constraints& constraints) {
    Kokkos::parallel_for(
        "UpdateAlgorithmicAcceleration", solver.num_system_nodes,
        UpdateAlgorithmicAcceleration{
            state.a,
            state.vd,
            parameters.alpha_f,
            parameters.alpha_m,
        }
    );

    Kokkos::parallel_for(
        "UpdateGlobalPosition", solver.num_system_nodes,
        UpdateGlobalPosition{
            state.q,
            state.x0,
            state.x,
        }
    );

    Kokkos::parallel_for(
        "CalculateConstraintOutput", constraints.num,
        CalculateConstraintOutput{
            constraints.type,
            constraints.target_node_index,
            constraints.axes,
            state.x0,
            state.q,
            state.v,
            state.vd,
            constraints.output,
        }
    );
}
}
