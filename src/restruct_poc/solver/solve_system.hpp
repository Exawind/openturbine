#pragma once

#include <KokkosBlas.hpp>
#include <KokkosLapack_gesv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "condition_system.hpp"
#include "solver.hpp"

namespace openturbine {

inline void SolveSystem(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");
    
    {
        auto assemble_region = Kokkos::Profiling::ScopedRegion("Assemble Full System");
        KokkosSparse::spadd_numeric(
            &solver.spc_spadd_handle, solver.conditioner, solver.system_matrix_full, 1.,
            solver.constraints_matrix_full, solver.system_plus_constraints
        );
        KokkosSparse::spadd_numeric(
            &solver.full_system_spadd_handle, 1., solver.system_plus_constraints, 1.,
            solver.transpose_matrix_full, solver.full_matrix
        );
    }

    Kokkos::parallel_for(
        "ConditionR", solver.num_system_dofs,
        ConditionR{
            solver.R,
            solver.conditioner,
        }
    );

    KokkosBlas::axpby(
        -1.0, solver.R, 0.0,
        Kokkos::subview(solver.b->getLocalViewDevice(Tpetra::Access::OverwriteAll), Kokkos::ALL(), 0)
    );

    Kokkos::deep_copy(solver.A->getLocalMatrixDevice().values, solver.full_matrix.values);

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
            solver.conditioner,
            solver.x,
        }
    );
}

}  // namespace openturbine
