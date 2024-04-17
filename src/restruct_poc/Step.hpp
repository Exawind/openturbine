#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"
#include "beams.hpp"
#include "UpdateAlgorithmicAcceleration.hpp"
#include "AssembleSystem.hpp"
#include "AssembleConstraints.hpp"
#include "SolveSystem.hpp"
#include "CalculateConvergenceError.hpp"
#include "UpdateStatePrediction.hpp"
#include "PredictNextState.hpp"
#include "UpdateState.hpp"

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

    auto St_11 = Kokkos::subview(solver.St, system_range, system_range);
    auto St_12 = Kokkos::subview(solver.St, system_range, constraint_range);
    auto St_21 = Kokkos::subview(solver.St, constraint_range, system_range);

    solver.convergence_err.clear();

    double err = 1000.0;

    for (int iter = 0; err > 1.0; ++iter) {
        Kokkos::deep_copy(solver.St, 0.);

        UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);

        AssembleSystem(solver, beams, St_11, R_system);

        AssembleConstraints(solver, St_12, St_21, R_system, R_lambda);

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

}
