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

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {

template <typename Subview_NxN, typename Subview_N>
void AssembleConstraints(
    Solver& solver, Beams& beams, Subview_NxN St_12, Subview_NxN St_21, Subview_N R_system,
    Subview_N R_lambda
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

    auto B_num_rows = solver.constraints.B.extent(0);
    auto B_num_columns = solver.constraints.B.extent(1);

    auto B_row_data_size = Kokkos::View<double*>::shmem_size(B_num_columns);
    auto B_col_idx_size = Kokkos::View<int*>::shmem_size(B_num_columns);
    auto constraint_policy = Kokkos::TeamPolicy<>(B_num_rows, Kokkos::AUTO());
    constraint_policy.set_scratch_size(1, Kokkos::PerTeam(B_row_data_size + B_col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", constraint_policy, CopyIntoSparseMatrix{solver.B, solver.constraints.B}
    );

    auto B_t_row_data_size = Kokkos::View<double*>::shmem_size(B_num_rows);
    auto B_t_col_idx_size = Kokkos::View<int*>::shmem_size(B_num_rows);
    auto constraint_transpose_policy = Kokkos::TeamPolicy<>(B_num_columns, Kokkos::AUTO());
    constraint_transpose_policy.set_scratch_size(1, Kokkos::PerTeam(B_t_row_data_size + B_t_col_idx_size));
    Kokkos::parallel_for(
        "CopyIntoSparseMatrix_Transpose", constraint_transpose_policy, CopyIntoSparseMatrix_Transpose{solver.B_t, solver.constraints.B}
    );

    auto R = Kokkos::View<double*>("R_local", R_system.extent(0));
    Kokkos::deep_copy(R, R_system);
    auto spmv_handle = Solver::SpmvHandle();
    KokkosSparse::spmv(&spmv_handle, "T", 1., solver.B, solver.state.lambda, 1., R);
    Kokkos::deep_copy(R_system, R);

    Kokkos::deep_copy(R_lambda, solver.constraints.Phi);

    auto B_t = solver.B_t;
    auto transpose_copy_policy = Kokkos::TeamPolicy<>(St_12.extent(0), Kokkos::AUTO());
    Kokkos::parallel_for(
        "Copy into St_12", transpose_copy_policy,
        KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
            auto i = member.league_rank();
            auto row = B_t.row(i);
            auto row_map = B_t.graph.row_map;
            auto cols = B_t.graph.entries;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
                St_12(i, cols(row_map(i) + entry)) = row.value(entry);
            });
        }
    );

    Kokkos::fence();
    auto constraints_spgemm_handle = Solver::KernelHandle();
    constraints_spgemm_handle.create_spgemm_handle();
    KokkosSparse::spgemm_symbolic(constraints_spgemm_handle, solver.B, false, solver.T, false, solver.constraints_matrix);
    KokkosSparse::spgemm_numeric(constraints_spgemm_handle, solver.B, false, solver.T, false, solver.constraints_matrix);

    Kokkos::fence();
    auto copy_policy = Kokkos::TeamPolicy<>(St_21.extent(0), Kokkos::AUTO());
    auto constraints_matrix = solver.constraints_matrix;
    Kokkos::parallel_for(
        "Copy into St_21", copy_policy,
        KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
            auto i = member.league_rank();
            auto row = constraints_matrix.row(i);
            auto row_map = constraints_matrix.graph.row_map;
            auto cols = constraints_matrix.graph.entries;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
                St_21(i, cols(row_map(i) + entry)) = row.value(entry);
            });
        }
    );
}

}  // namespace openturbine
