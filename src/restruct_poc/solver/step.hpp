#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"
#include "assemble_constraints_matrix.hpp"
#include "assemble_constraints_residual.hpp"
#include "assemble_system_matrix.hpp"
#include "assemble_system_residual.hpp"
#include "assemble_tangent_operator.hpp"
#include "calculate_convergence_error.hpp"
#include "predict_next_state.hpp"
#include "solve_system.hpp"
#include "update_algorithmic_acceleration.hpp"
#include "update_constraint_variables.hpp"
#include "update_state_prediction.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/update_state.hpp"

namespace openturbine {

inline bool Step(Solver& solver, Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Step");
    PredictNextState(solver);

    solver.convergence_err.clear();

    double err = 1000.0;

    auto beta_prime = (solver.is_dynamic_solve) ? solver.beta_prime : 0.;
    auto gamma_prime = (solver.is_dynamic_solve) ? solver.gamma_prime : 0.;
    for (auto iter = 0U; err > 1.0; ++iter) {
        UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd, beta_prime, gamma_prime);

        AssembleTangentOperator(solver);

        AssembleSystemResidual(solver, beams);

        AssembleSystemMatrix(solver, beams);

        UpdateConstraintVariables(solver);

        AssembleConstraintsMatrix(solver);

        AssembleConstraintsResidual(solver);

        SolveSystem(solver);

        err = CalculateConvergenceError(solver);

        solver.convergence_err.push_back(err);

        UpdateStatePrediction(solver);

        if (iter >= solver.max_iter) {
            return false;
        }
    }

    Kokkos::parallel_for(
        "UpdateAlgorithmicAcceleration", solver.num_system_nodes,
        UpdateAlgorithmicAcceleration{
            solver.state.a,
            solver.state.vd,
            solver.alpha_f,
            solver.alpha_m,
        }
    );

    return true;
}

}  // namespace openturbine
