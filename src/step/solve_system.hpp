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
        ConditionR{parameters.conditioner, solver.b->getLocalViewDevice(Tpetra::Access::ReadWrite)}
    );

    solver.b->scale(-1.);

    {
        auto solve_region = Kokkos::Profiling::ScopedRegion("Linear Solve");
        solver.amesos_solver->numericFactorization();
        solver.amesos_solver->solve();
    }

    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs - solver.num_system_dofs,
        UnconditionSolution{
            solver.num_system_dofs,
            parameters.conditioner,
            solver.x->getLocalViewDevice(Tpetra::Access::ReadWrite),
        }
    );
}

}  // namespace openturbine
