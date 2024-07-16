#pragma once

#include <array>
#include <vector>

#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>

#include "compute_number_of_non_zeros.hpp"
#include "constraint.hpp"
#include "constraints.hpp"
#include "fill_unshifted_row_ptrs.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_indices_constraints.hpp"
#include "populate_sparse_indices_constraints_transpose.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "populate_sparse_row_ptrs_constraints.hpp"
#include "populate_sparse_row_ptrs_constraints_transpose.hpp"
#include "populate_tangent_indices.hpp"
#include "populate_tangent_row_ptrs.hpp"
#include "state.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Solver {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using CrsMatrixType = KokkosSparse::CrsMatrix<double, int, DeviceType, void, int>;
    using DenseMatrixType = Kokkos::View<double**, Kokkos::LayoutLeft>;
    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<
        int, int, double, ExecutionSpace, MemorySpace, MemorySpace>;
    using SpmvHandle = KokkosSparse::SPMVHandle<
        ExecutionSpace, CrsMatrixType, Kokkos::View<double*>, Kokkos::View<double*>>;
    using ValuesType = Kokkos::View<CrsMatrixType::value_type*>;
    using RowPtrType = Kokkos::View<CrsMatrixType::size_type*>;
    using IndicesType = Kokkos::View<CrsMatrixType::ordinal_type*>;
    bool is_dynamic_solve;
    int max_iter;
    double h;
    double alpha_m;
    double alpha_f;
    double gamma;
    double beta;
    double gamma_prime;
    double beta_prime;
    double conditioner;
    int num_system_nodes;
    int num_system_dofs;
    Constraints constraints;
    int num_dofs;
    State state;
    CrsMatrixType K;
    CrsMatrixType T;
    CrsMatrixType B;
    CrsMatrixType B_t;
    CrsMatrixType static_system_matrix;
    CrsMatrixType system_matrix;
    CrsMatrixType constraints_matrix;
    CrsMatrixType system_matrix_full;
    CrsMatrixType constraints_matrix_full;
    CrsMatrixType transpose_matrix_full;
    CrsMatrixType system_plus_constraints;
    CrsMatrixType full_matrix;
    View_NxN K_dense;  // Stiffness matrix
    View_NxN St;       // Iteration matrix
    DenseMatrixType St_left;
    Kokkos::View<int*, Kokkos::LayoutLeft> IPIV;
    View_N R;  // System residual vector
    View_N x;  // System solution vector

    std::vector<double> convergence_err;

    KernelHandle system_spgemm_handle;
    KernelHandle constraints_spgemm_handle;
    KernelHandle system_spadd_handle;
    KernelHandle spc_spadd_handle;
    KernelHandle full_system_spadd_handle;

    Solver(
        bool is_dynamic_solve_, int max_iter_, double h_, double rho_inf,
        std::vector<Node>& system_nodes, std::vector<Constraint> constraints_, Beams& beams_
    )
        : is_dynamic_solve(is_dynamic_solve_),
          max_iter(max_iter_),
          h(h_),
          alpha_m((2. * rho_inf - 1.) / (rho_inf + 1.)),
          alpha_f(rho_inf / (rho_inf + 1.)),
          gamma(0.5 + alpha_f - alpha_m),
          beta(0.25 * (gamma + 0.5) * (gamma + 0.5)),
          gamma_prime(gamma / (h * beta)),
          beta_prime((1. - alpha_m) / (h * h * beta * (1. - alpha_f))),
          conditioner(beta * h * h),
          num_system_nodes(system_nodes.size()),
          num_system_dofs(num_system_nodes * kLieAlgebraComponents),
          constraints(constraints_, num_system_dofs),
          num_dofs(num_system_dofs + constraints.num_dofs),
          state(num_system_nodes, constraints.num_dofs, system_nodes),
          K_dense("K dense", num_system_dofs, num_system_dofs),
          St("St", num_dofs, num_dofs),
          IPIV("IPIV", num_dofs),
          R("R", num_dofs),
          x("x", num_dofs),
          convergence_err(max_iter),
          system_spgemm_handle(),
          constraints_spgemm_handle(),
          system_spadd_handle() {
        if constexpr (!std::is_same_v<decltype(St)::array_layout, Kokkos::LayoutLeft>) {
            St_left = Kokkos::View<double**, Kokkos::LayoutLeft>("St_left", num_dofs, num_dofs);
        }

        auto K_num_rows = this->num_system_dofs;
        auto K_num_columns = this->num_system_dofs;
        auto K_num_non_zero = 0;
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros", beams_.num_elems,
            ComputeNumberOfNonZeros{beams_.elem_indices}, K_num_non_zero
        );
        auto K_row_ptrs = Kokkos::View<int*>("K_row_ptrs", K_num_rows + 1);
        auto K_col_inds = Kokkos::View<int*>("indices", K_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrs", 1, PopulateSparseRowPtrs{beams_.elem_indices, K_row_ptrs}
        );
        Kokkos::parallel_for(
            "PopulateSparseIndices", 1,
            PopulateSparseIndices{beams_.elem_indices, beams_.node_state_indices, K_col_inds}
        );

        Kokkos::fence();
        auto K_values = Kokkos::View<double*>("K values", K_num_non_zero);
        K = CrsMatrixType(
            "K", K_num_rows, K_num_columns, K_num_non_zero, K_values, K_row_ptrs, K_col_inds
        );

        // Tangent operator sparse matrix
        auto T_num_non_zero = this->num_system_nodes * 6 * 6;
        auto T_row_ptrs = RowPtrType("T_row_ptrs", K_num_rows + 1);
        auto T_indices = IndicesType("T_indices", T_num_non_zero);
        Kokkos::parallel_for(
            "PopulateTangentRowPtrs", 1,
            PopulateTangentRowPtrs<CrsMatrixType::size_type>{this->num_system_nodes, T_row_ptrs}
        );
        auto node_ids = Kokkos::View<int*>("node_ids", system_nodes.size());
        auto host_node_ids = Kokkos::create_mirror(node_ids);
        for (auto i = 0u; i < system_nodes.size(); ++i) {
            host_node_ids(i) = system_nodes[i].ID;
        }
        Kokkos::deep_copy(node_ids, host_node_ids);
        Kokkos::parallel_for(
            "PopulateTangentIndices", 1,
            PopulateTangentIndices{this->num_system_nodes, node_ids, T_indices}
        );
        auto T_values = ValuesType("T values", T_num_non_zero);
        T = CrsMatrixType(
            "T", K_num_rows, K_num_columns, T_num_non_zero, T_values, T_row_ptrs, T_indices
        );

        // Initialize contraint for indexing for sparse matrices
        int B_num_non_zero = 0;
        for (const auto& constraint : constraints_) {
            B_num_non_zero += (constraint.base_node.ID < 0 ? 1 : 2) * 6 * 6;
        }
        auto B_num_rows = this->constraints.num_dofs;
        auto B_num_columns = this->num_system_dofs;
        auto B_row_ptrs = Kokkos::View<int*>("b_row_ptrs", B_num_rows + 1);
        auto B_col_ind = Kokkos::View<int*>("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrs_Constraints", 1,
            PopulateSparseRowPtrs_Constraints{
                this->constraints.num, this->constraints.data, B_row_ptrs}
        );

        Kokkos::parallel_for(
            "PopulateSparseIndices_Constraints", 1,
            PopulateSparseIndices_Constraints{
                this->constraints.num, this->constraints.data, B_col_ind}
        );

        auto B_values = Kokkos::View<double*>("B values", B_num_non_zero);
        KokkosSparse::sort_crs_matrix(B_row_ptrs, B_col_ind, B_values);
        B = CrsMatrixType(
            "B", B_num_rows, B_num_columns, B_num_non_zero, B_values, B_row_ptrs, B_col_ind
        );

        auto B_t_num_rows = B_num_columns;
        auto B_t_num_columns = B_num_rows;
        auto B_t_num_non_zero = B_num_non_zero;
        auto B_t_row_ptrs = Kokkos::View<int*>("b_t_row_ptrs", B_t_num_rows + 1);
        auto B_t_indices = Kokkos::View<int*>("B_t_indices", B_t_num_non_zero);

        Kokkos::parallel_for(
            "PopulateSparseRowPtrs_Constraints_Transpose", 1,
            PopulateSparseRowPtrs_Constraints_Transpose{
                this->constraints.num, this->num_system_nodes, this->constraints.data, B_t_row_ptrs}
        );
        Kokkos::parallel_for(
            "PopulateSparseIndices_Constraints_Transpose", 1,
            PopulateSparseIndices_Constraints_Transpose{
                this->constraints.num, this->num_system_nodes, this->constraints.data, B_t_indices}
        );

        auto B_t_values = Kokkos::View<double*>("B_t values", B_t_num_non_zero);
        KokkosSparse::sort_crs_matrix(B_t_row_ptrs, B_t_indices, B_t_values);
        B_t = CrsMatrixType(
            "B_t", B_t_num_rows, B_t_num_columns, B_t_num_non_zero, B_t_values, B_t_row_ptrs,
            B_t_indices
        );

        system_spgemm_handle.create_spgemm_handle();
        KokkosSparse::spgemm_symbolic(
            system_spgemm_handle, K, false, T, false, static_system_matrix
        );

        constraints_spgemm_handle.create_spgemm_handle();
        KokkosSparse::spgemm_symbolic(
            constraints_spgemm_handle, B, false, T, false, constraints_matrix
        );

        system_spadd_handle.create_spadd_handle(true);
        KokkosSparse::spadd_symbolic(&system_spadd_handle, K, static_system_matrix, system_matrix);

        auto system_matrix_full_row_ptrs =
            Kokkos::View<int*>("system_matrix_full_row_ptrs", num_dofs + 1);
        Kokkos::parallel_for(
            "FillUnshiftedRowPtrs", num_dofs + 1,
            FillUnshiftedRowPtrs{
                system_matrix_full_row_ptrs, num_system_dofs, system_matrix.graph.row_map}
        );
        system_matrix_full = CrsMatrixType(
            "system_matrix_full", num_dofs, num_dofs, system_matrix.nnz(), system_matrix.values,
            system_matrix_full_row_ptrs, system_matrix.graph.entries
        );

        auto constraints_matrix_full_row_ptrs =
            Kokkos::View<int*>("constraints_matrix_full_row_ptrs", num_dofs + 1);
        Kokkos::deep_copy(
            Kokkos::subview(
                constraints_matrix_full_row_ptrs, Kokkos::pair(num_system_dofs, num_dofs + 1)
            ),
            constraints_matrix.graph.row_map
        );
        constraints_matrix_full = CrsMatrixType(
            "constraints_matrix_full", num_dofs, num_dofs, constraints_matrix.nnz(),
            constraints_matrix.values, constraints_matrix_full_row_ptrs,
            constraints_matrix.graph.entries
        );

        auto transpose_matrix_full_row_ptrs =
            Kokkos::View<int*>("transpose_matrix_full_row_ptrs", num_dofs + 1);
        Kokkos::parallel_for(
            "FillUnshiftedRowPtrs", num_dofs + 1,
            FillUnshiftedRowPtrs{transpose_matrix_full_row_ptrs, num_system_dofs, B_t.graph.row_map}
        );
        auto transpose_matrix_full_indices =
            Kokkos::View<int*>("transpose_matrix_full_indices", B_t.nnz());
        Kokkos::deep_copy(transpose_matrix_full_indices, num_system_dofs);
        KokkosBlas::axpy(1., B_t.graph.entries, transpose_matrix_full_indices);
        transpose_matrix_full = CrsMatrixType(
            "transpose_matrix_full", num_dofs, num_dofs, B_t.nnz(), B_t.values,
            transpose_matrix_full_row_ptrs, transpose_matrix_full_indices
        );

        spc_spadd_handle.create_spadd_handle(true);
        KokkosSparse::spadd_symbolic(
            &spc_spadd_handle, system_matrix_full, constraints_matrix_full, system_plus_constraints
        );

        full_system_spadd_handle.create_spadd_handle(true);
        KokkosSparse::spadd_symbolic(
            &full_system_spadd_handle, system_plus_constraints, transpose_matrix_full, full_matrix
        );
    }
};

}  // namespace openturbine
