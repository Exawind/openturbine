#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "CalculateConstraintResidualGradient.hpp"
#include "Solver.hpp"
#include "UpdateIterationMatrix.hpp"

#include "src/restruct_poc/beams/Beams.hpp"

namespace openturbine {

template <typename Subview_NxN, typename Subview_N>
void AssembleConstraints(
    Solver& solver, Subview_NxN St_12, Subview_NxN St_21, Subview_N R_system, Subview_N R_lambda
) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Constraints");
    if (solver.num_constraint_dofs == 0) {
        return;
    }

    Kokkos::deep_copy(solver.constraints.Phi, 0.);
    Kokkos::deep_copy(solver.constraints.B, 0.);
    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", solver.num_constraint_nodes,
        CalculateConstraintResidualGradient{
            solver.constraints.node_indices,
            solver.constraints.X0,
            solver.constraints.u,
            solver.state.q,
            solver.constraints.Phi,
            solver.constraints.B,
        }
    );

    KokkosBlas::gemv("T", 1.0, solver.constraints.B, solver.state.lambda, 1.0, R_system);
    Kokkos::deep_copy(R_lambda, solver.constraints.Phi);

    Kokkos::parallel_for(
        "UpdateIterationMatrix",
        Kokkos::MDRangePolicy{{0, 0}, {solver.num_constraint_dofs, solver.num_system_dofs}},
        UpdateIterationMatrix<Subview_NxN>{St_12, solver.constraints.B}
    );

    KokkosBlas::gemm("N", "N", 1.0, solver.constraints.B, solver.T, 0.0, St_21);
}

}  // namespace openturbine
