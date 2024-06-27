#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_constraints.hpp"
#include "assemble_system.hpp"
#include "calculate_convergence_error.hpp"
#include "predict_next_state.hpp"
#include "solve_system.hpp"
#include "solver.hpp"
#include "update_algorithmic_acceleration.hpp"
#include "update_state_prediction.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/update_state.hpp"

namespace openturbine {

inline bool Step(Solver& solver, Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Step");
    PredictNextState(solver);

    auto system_range = Kokkos::make_pair(0, solver.num_system_dofs);
    auto constraint_range = Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs);

    auto R_system = Kokkos::subview(solver.R, system_range);
    auto R_lambda = Kokkos::subview(solver.R, constraint_range);

    auto x_system = Kokkos::subview(solver.x, system_range);
    auto x_lambda = Kokkos::subview(solver.x, constraint_range);

    solver.convergence_err.clear();

    double err = 1000.0;

    for (int iter = 0; err > 1.0; ++iter) {
        UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);

        AssembleSystem(solver, beams, R_system);

        AssembleConstraints(solver, R_system, R_lambda);

        SolveSystem(solver);

        err = CalculateConvergenceError(solver);

        solver.convergence_err.push_back(err);

        UpdateStatePrediction(solver, x_system, x_lambda);

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
