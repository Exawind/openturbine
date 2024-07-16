#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_constraint_residual_gradient.hpp"
#include "compute_number_of_non_zeros.hpp"
#include "copy_into_sparse_matrix.hpp"
#include "copy_into_sparse_matrix_transpose.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "solver.hpp"
#include "update_iteration_matrix.hpp"

namespace openturbine {

template <typename Subview_N>
void AssembleConstraints(Solver& solver, Subview_N R_system, Subview_N R_lambda) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Constraints");

    // If no constraints, return
    if (solver.constraints.num == 0) {
        return;
    }

    // Transfer prescribed displacements to host
    solver.constraints.UpdateViews();

    Kokkos::deep_copy(solver.constraints.Phi, 0.);
    Kokkos::deep_copy(solver.constraints.B, 0.);
    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", solver.constraints.num,
        CalculateConstraintResidualGradient{
            solver.constraints.data,
            solver.constraints.control,
            solver.constraints.u,
            solver.state.q,
            solver.constraints.Phi,
            solver.constraints.B,
        }
    );

    auto B_num_rows = solver.constraints.B.extent(0);
    auto B_num_columns = solver.constraints.B.extent(1);

    auto B_row_data_size = Kokkos::View<double*>::shmem_size(B_num_columns);
    auto B_col_idx_size = Kokkos::View<int*>::shmem_size(B_num_columns);
    auto constraint_policy = Kokkos::TeamPolicy<>(B_num_rows, Kokkos::AUTO());
    constraint_policy.set_scratch_size(1, Kokkos::PerTeam(B_row_data_size + B_col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", constraint_policy,
        CopyIntoSparseMatrix{solver.B, solver.constraints.B}
    );

    auto B_t_row_data_size = Kokkos::View<double*>::shmem_size(B_num_rows);
    auto B_t_col_idx_size = Kokkos::View<int*>::shmem_size(B_num_rows);
    auto constraint_transpose_policy = Kokkos::TeamPolicy<>(B_num_columns, Kokkos::AUTO());
    constraint_transpose_policy.set_scratch_size(
        1, Kokkos::PerTeam(B_t_row_data_size + B_t_col_idx_size)
    );
    Kokkos::parallel_for(
        "CopyIntoSparseMatrix_Transpose", constraint_transpose_policy,
        CopyIntoSparseMatrix_Transpose{solver.B_t, solver.constraints.B}
    );

    {
        auto resid_region = Kokkos::Profiling::ScopedRegion("Assemble Residual");
        auto R = Kokkos::View<double*>("R_local", R_system.extent(0));
        Kokkos::deep_copy(R, R_system);
        auto spmv_handle = Solver::SpmvHandle();
        KokkosSparse::spmv(&spmv_handle, "T", 1., solver.B, solver.state.lambda, 1., R);
        Kokkos::deep_copy(R_system, R);
        Kokkos::deep_copy(R_lambda, solver.constraints.Phi);
    }

    Kokkos::fence();
    {
        auto mult_region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Matrix");
        KokkosSparse::spgemm_numeric(
            solver.constraints_spgemm_handle, solver.B, false, solver.T, false,
            solver.constraints_matrix
        );
    }
}

}  // namespace openturbine
