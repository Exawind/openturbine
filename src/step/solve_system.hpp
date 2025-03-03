#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver/condition_system.hpp"
#include "solver/solver.hpp"
#include "step_parameters.hpp"

namespace openturbine {

inline void SolveSystem(StepParameters& parameters, Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");

    Kokkos::parallel_for(
        "ConditionR", solver.num_system_dofs,
        ConditionR{
            solver.R,
            parameters.conditioner,
        }
    );

    KokkosBlas::axpby(
        -1.0, solver.R, 0.0,
        Kokkos::subview(solver.b->getLocalViewDevice(Tpetra::Access::OverwriteAll), Kokkos::ALL(), 0)
    );

    {
        auto solve_region = Kokkos::Profiling::ScopedRegion("Linear Solve");
        solver.amesos_solver->numericFactorization();
        solver.amesos_solver->solve();
    }

    Kokkos::deep_copy(
        solver.x,
        Kokkos::subview(solver.x_mv->getLocalViewDevice(Tpetra::Access::ReadOnly), Kokkos::ALL(), 0)
    );

    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs,
        UnconditionSolution{
            solver.num_system_dofs,
            parameters.conditioner,
            solver.x,
        }
    );
}

}  // namespace openturbine
