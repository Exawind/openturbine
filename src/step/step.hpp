#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_constraints_matrix.hpp"
#include "assemble_constraints_residual.hpp"
#include "assemble_system_matrix.hpp"
#include "assemble_system_residual.hpp"
#include "assemble_tangent_operator.hpp"
#include "calculate_convergence_error.hpp"
#include "predict_next_state.hpp"
#include "reset_constraints.hpp"
#include "solve_system.hpp"
#include "step_parameters.hpp"
#include "update_constraint_prediction.hpp"
#include "update_constraint_variables.hpp"
#include "update_state_prediction.hpp"
#include "update_system_variables.hpp"
#include "update_tangent_operator.hpp"

#include "src/beams/beams.hpp"
#include "src/constraints/constraints.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/state/update_algorithmic_acceleration.hpp"
#include "src/state/update_global_position.hpp"

namespace openturbine {

inline bool Step(
    StepParameters& parameters, Solver& solver, Beams& beams, State& state, Constraints& constraints
) {
    auto region = Kokkos::Profiling::ScopedRegion("Step");
    PredictNextState(parameters, state);

    ResetConstraints(constraints);

    solver.convergence_err.clear();

    double err = 1000.0;

    for (auto iter = 0U; err > 1.0; ++iter) {
        UpdateSystemVariables(parameters, beams, state);

        UpdateTangentOperator(parameters, state);

        AssembleTangentOperator(solver, state);

        AssembleSystemResidual(solver, beams);

        AssembleSystemMatrix(solver, beams);

        UpdateConstraintVariables(state, constraints);

        AssembleConstraintsMatrix(solver, constraints);

        AssembleConstraintsResidual(solver, constraints);

        SolveSystem(parameters, solver);

        err = CalculateConvergenceError(solver, state);

        solver.convergence_err.push_back(err);

        UpdateStatePrediction(parameters, solver, state);

        UpdateConstraintPrediction(solver, constraints);

        if (iter >= parameters.max_iter) {
            return false;
        }
    }

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

    return true;
}

}  // namespace openturbine
