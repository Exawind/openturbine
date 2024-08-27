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
#include "solve_system.hpp"
#include "update_algorithmic_acceleration.hpp"
#include "update_constraint_variables.hpp"
#include "update_state_prediction.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/update_state.hpp"

namespace openturbine {

inline bool Step(Solver& solver, Beams& beams, State& state, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Step");
    PredictNextState(solver, state);

    solver.convergence_err.clear();

    double err = 1000.0;

    auto beta_prime = (solver.is_dynamic_solve) ? solver.beta_prime : 0.;
    auto gamma_prime = (solver.is_dynamic_solve) ? solver.gamma_prime : 0.;
    for (auto iter = 0U; err > 1.0; ++iter) {
        UpdateState(beams, state.q, state.v, state.vd, beta_prime, gamma_prime);

        AssembleTangentOperator(solver, state);

        AssembleSystemResidual(solver, beams);

        AssembleSystemMatrix(solver, beams);

        UpdateConstraintVariables(state, constraints);

        AssembleConstraintsMatrix(solver, constraints);

        AssembleConstraintsResidual(solver, state, constraints);

        SolveSystem(solver);

        err = CalculateConvergenceError(solver, state);

        solver.convergence_err.push_back(err);

        UpdateStatePrediction(solver, state, constraints);

        if (iter >= solver.max_iter) {
            return false;
        }
    }

    Kokkos::parallel_for(
        "UpdateAlgorithmicAcceleration", solver.num_system_nodes,
        UpdateAlgorithmicAcceleration{
            state.a,
            state.vd,
            solver.alpha_f,
            solver.alpha_m,
        }
    );

    return true;
}

}  // namespace openturbine
