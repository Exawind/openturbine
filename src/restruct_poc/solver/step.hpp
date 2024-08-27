#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"
#include "state.hpp"
#include "assemble_constraints_matrix.hpp"
#include "assemble_constraints_residual.hpp"
#include "assemble_system_matrix.hpp"
#include "assemble_system_residual.hpp"
#include "assemble_tangent_operator.hpp"
#include "calculate_convergence_error.hpp"
#include "constraints.hpp"
#include "predict_next_state.hpp"
#include "reset_constraints.hpp"
#include "step_parameters.hpp"
#include "solve_system.hpp"
#include "update_algorithmic_acceleration.hpp"
#include "update_constraint_variables.hpp"
#include "update_constraint_prediction.hpp"
#include "update_state_prediction.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/update_state.hpp"

namespace openturbine {

inline bool Step(StepParameters& parameters, Solver& solver, Beams& beams, State& state, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Step");
    PredictNextState(parameters, state);

    ResetConstraints(constraints);

    solver.convergence_err.clear();

    double err = 1000.0;

    for (auto iter = 0U; err > 1.0; ++iter) {
        UpdateState(parameters, beams, state);

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

    return true;
}

}  // namespace openturbine
