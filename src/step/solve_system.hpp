#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver/condition_system.hpp"
#include "solver/linear_solver/dss_numeric.hpp"
#include "solver/linear_solver/dss_solve.hpp"
#include "solver/solver.hpp"
#include "step_parameters.hpp"

namespace openturbine {

template <typename DeviceType>
inline void SolveSystem(StepParameters& parameters, Solver<DeviceType>& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");

    Kokkos::parallel_for(
        "ConditionR", solver.num_system_dofs, ConditionR{parameters.conditioner, solver.b}
    );

    KokkosBlas::scal(solver.b, -1., solver.b);

    {
        auto solve_region = Kokkos::Profiling::ScopedRegion("Linear Solve");
        dss_numeric(solver.handle, solver.A);
        dss_solve(solver.handle, solver.A, solver.b, solver.x);
    }

    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs - solver.num_system_dofs,
        UnconditionSolution{solver.num_system_dofs, parameters.conditioner, solver.x}
    );
}

}  // namespace openturbine
