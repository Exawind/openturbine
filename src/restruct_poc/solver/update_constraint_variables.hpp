#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_constraint_residual_gradient.hpp"
#include "solver.hpp"
#include "state.hpp"

namespace openturbine {

inline void UpdateConstraintVariables(Solver& solver, State& state) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Variables");

    if (solver.constraints.num == 0) {
        return;
    }

    solver.constraints.UpdateViews();

    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", solver.constraints.num,
        CalculateConstraintResidualGradient{
            solver.constraints.data, solver.constraints.control, solver.constraints.u,
            state.q, solver.constraints.Phi, solver.constraints.gradient_terms}
    );
}

}  // namespace openturbine
