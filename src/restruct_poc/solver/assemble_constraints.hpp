#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_constraint_residual_gradient.hpp"
#include "compute_number_of_non_zeros.hpp"
#include "copy_into_sparse_matrix.hpp"
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

    auto T_num_rows = solver.T.extent(0);
    auto T_num_columns = solver.T.extent(1);
    auto T_num_non_zero = 0;

    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros", beams.num_elems, ComputeNumberOfNonZeros{beams.elem_indices},
        T_num_non_zero
    );
    auto T_row_ptrs = Kokkos::View<int*>("row_ptrs", T_num_rows + 1);
    auto T_indices = Kokkos::View<int*>("indices", T_num_non_zero);
    Kokkos::parallel_for(
        "PopulateSparseRowPtrs", 1, PopulateSparseRowPtrs{beams.elem_indices, T_row_ptrs}
    );
    Kokkos::parallel_for(
        "PopulateSparseIndices", 1,
        PopulateSparseIndices{beams.elem_indices, beams.node_state_indices, T_indices}
    );

    using crs_matrix_type = KokkosSparse::CrsMatrix<
        double, int,
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>,
        void, int>;
    Kokkos::fence();
    auto T_values = Kokkos::View<double*>("T values", T_num_non_zero);
    auto T = crs_matrix_type(
        "T", T_num_rows, T_num_columns, T_num_non_zero, T_values, T_row_ptrs, T_indices
    );

    auto row_data_size = Kokkos::View<double*>::shmem_size(T_num_columns);
    auto col_idx_size = Kokkos::View<int*>::shmem_size(T_num_columns);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(T_num_rows, Kokkos::AUTO());
    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", sparse_matrix_policy, CopyIntoSparseMatrix{T, solver.T}
    );

    auto B_num_rows = solver.constraints.B.extent(0);
    auto B_num_columns = solver.constraints.B.extent(1);
    auto B_num_non_zero = solver.num_constraint_nodes * 6 * 6;

    auto B_row_ptrs = Kokkos::View<int*>("row_ptrs", B_num_rows + 1);
    auto B_indices = Kokkos::View<int*>("indices", B_num_non_zero);

    auto num_constraint_nodes = solver.num_constraint_nodes;
    auto node_indices = solver.constraints.node_indices;
    Kokkos::parallel_for(
        "PopulateSparseRowPtrs_Constraints", 1,
        KOKKOS_LAMBDA(int) {
            auto rows_so_far = 0;
            for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
                for (int i = 0; i < kLieAlgebraComponents; ++i) {
                    B_row_ptrs(rows_so_far + 1) = B_row_ptrs(rows_so_far) + kLieAlgebraComponents;
                    ++rows_so_far;
                }
            }
        }
    );

    Kokkos::parallel_for(
        "PopulateSparseIndices_Constraints", 1,
        KOKKOS_LAMBDA(int) {
            auto entries_so_far = 0;
            for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
                auto i_node2 = node_indices(i_constraint).constrained_node_index;
                auto i_col = i_node2 * kLieAlgebraComponents;
                for (int i = 0; i < kLieAlgebraComponents; ++i) {
                    for (int j = 0; j < kLieAlgebraComponents; ++j) {
                        B_indices(entries_so_far) = i_col + j;
                        ++entries_so_far;
                    }
                }
            }
        }
    );

    auto B_values = Kokkos::View<double*>("B values", B_num_non_zero);
    auto B = crs_matrix_type(
        "B", B_num_rows, B_num_columns, B_num_non_zero, B_values, B_row_ptrs, B_indices
    );

    auto B_row_data_size = Kokkos::View<double*>::shmem_size(B_num_columns);
    auto B_col_idx_size = Kokkos::View<int*>::shmem_size(B_num_columns);
    auto constraint_policy = Kokkos::TeamPolicy<>(B_num_rows, Kokkos::AUTO());
    constraint_policy.set_scratch_size(1, Kokkos::PerTeam(B_row_data_size + B_col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", constraint_policy, CopyIntoSparseMatrix{B, solver.constraints.B}
    );

    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<
        int, int, double, Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::DefaultExecutionSpace::memory_space>;
    KernelHandle kh;
    kh.create_spgemm_handle();

    auto spmv_handle = KokkosSparse::SPMVHandle<
        Kokkos::DefaultExecutionSpace, decltype(B), decltype(solver.state.lambda),
        decltype(R_system)>();
    KokkosSparse::spmv(&spmv_handle, "T", 1., B, solver.state.lambda, 1., R_system);

    Kokkos::deep_copy(R_lambda, solver.constraints.Phi);

    auto transpose_copy_policy = Kokkos::TeamPolicy<>(St_12.extent(1), Kokkos::AUTO());
    Kokkos::parallel_for(
        "Copy into St_12", transpose_copy_policy,
        KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
            auto i = member.league_rank();
            auto row = B.row(i);
            auto row_map = B.graph.row_map;
            auto cols = B.graph.entries;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
                St_12(cols(row_map(i) + entry), i) = row.value(entry);
            });
        }
    );

    Kokkos::fence();
    crs_matrix_type system_matrix;
    KokkosSparse::spgemm_symbolic(kh, B, false, T, false, system_matrix);
    KokkosSparse::spgemm_numeric(kh, B, false, T, false, system_matrix);

    Kokkos::fence();
    auto copy_policy = Kokkos::TeamPolicy<>(St_21.extent(0), Kokkos::AUTO());
    Kokkos::parallel_for(
        "Copy into St_21", copy_policy,
        KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
            auto i = member.league_rank();
            auto row = system_matrix.row(i);
            auto row_map = system_matrix.graph.row_map;
            auto cols = system_matrix.graph.entries;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
                St_21(i, cols(row_map(i) + entry)) = row.value(entry);
            });
        }
    );
}

}  // namespace openturbine
