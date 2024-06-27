#pragma once

#include <array>
#include <vector>

#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>

#include "compute_number_of_non_zeros.hpp"
#include "compute_number_of_non_zeros_constraints.hpp"
#include "constraint.hpp"
#include "constraints.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "populate_sparse_row_ptrs_col_inds_constraints.hpp"
#include "populate_sparse_row_ptrs_col_inds_transpose.hpp"
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
    CrsMatrixType system_plus_constraints;
    CrsMatrixType full_matrix;
    View_NxN K_dense;  // Stiffness matrix
    View_NxN St;       // Iteration matrix
    DenseMatrixType St_left;
    Kokkos::View<int*, Kokkos::LayoutLeft> IPIV;
    View_Nx6x6 T_dense;
    Kokkos::View<double***> matrix_terms;
    View_N R;  // System residual vector
    View_N x;  // System solution vector

    std::vector<double> convergence_err;

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
          num_constraint_nodes(constraint_inputs.size()),
          num_constraint_dofs(num_constraint_nodes * kLieAlgebraComponents),
          T_dense("T dense", num_system_nodes),
          matrix_terms("matrix_terms", beams_.num_elems, beams_.max_elem_nodes*kLieAlgebraComponents, beams_.max_elem_nodes*kLieAlgebraComponents),
          R("R", num_dofs),
          x("x", num_dofs),
          convergence_err(max_iter) {
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

        auto T_num_non_zero = num_system_nodes * 6 * 6;
        auto T_row_ptrs = RowPtrType("T_row_ptrs", num_rows + 1);
        auto T_indices = IndicesType("T_indices", T_num_non_zero);
        Kokkos::parallel_for("PopulateTangentRowPtrs", 1, PopulateTangentRowPtrs<CrsMatrixType::size_type>{beams_.elem_indices, T_row_ptrs});
        Kokkos::parallel_for(
            "PopulateTangentIndices", 1,
            PopulateTangentIndices{beams_.elem_indices, beams_.node_state_indices, T_indices}
        );
        auto T_values = ValuesType("T values", T_num_non_zero);
        T = CrsMatrixType("T", num_rows, num_columns, T_num_non_zero, T_values, T_row_ptrs, T_indices);

        auto B_num_rows = num_constraint_dofs;
        auto B_num_columns = num_system_dofs;
        auto B_num_non_zero = num_constraint_nodes * 6 * 6;
        auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
        auto B_indices = IndicesType("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrs_Constraints", 1,
            PopulateSparseRowPtrs_Constraints<CrsMatrixType::size_type>{
                num_constraint_nodes, B_row_ptrs}
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
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros_Constraints", this->constraints.num,
            ComputeNumberOfNonZeros_Constraints{this->constraints.data}, B_num_non_zero
        );
        auto B_num_rows = this->constraints.num_dofs;
        auto B_num_columns = this->num_system_dofs;
        auto B_row_ptrs = Kokkos::View<int*>("b_row_ptrs", B_num_rows + 1);
        auto B_col_ind = Kokkos::View<int*>("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrsColInds_Constraints", 1,
            PopulateSparseRowPtrsColInds_Constraints{this->constraints.data, B_row_ptrs, B_col_ind}
        );
        auto B_values = Kokkos::View<double*>("B values", B_num_non_zero);
        KokkosSparse::sort_crs_matrix(B_row_ptrs, B_col_ind, B_values);
        B = CrsMatrixType(
            "B", B_num_rows, B_num_columns, B_num_non_zero, B_values, B_row_ptrs, B_col_ind
        );

        auto B_t_num_rows = B_num_columns;
        auto B_t_num_columns = B_num_rows;
        auto B_t_num_non_zero = B_num_non_zero;
        auto col_count = Kokkos::View<int*>("col_count", B_num_columns);
        auto tmp_row_ptrs = Kokkos::View<int*>("tmp_row_ptrs", B_t_num_rows + 1);
        auto B_t_row_ptrs = Kokkos::View<int*>("b_t_row_ptrs", B_t_num_rows + 1);
        auto B_t_col_inds = Kokkos::View<int*>("B_t_indices", B_t_num_non_zero);
        auto B_t_values = Kokkos::View<double*>("B_t values", B_t_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrsColInds_Transpose", 1,
            PopulateSparseRowPtrsColInds_Transpose{
                B_num_rows, B_num_columns, B_row_ptrs, B_col_ind, col_count, tmp_row_ptrs,
                B_t_row_ptrs, B_t_col_inds}
        );
        B_t = CrsMatrixType(
            "B_t", B_t_num_rows, B_t_num_columns, B_t_num_non_zero, B_t_values, B_t_row_ptrs,
            B_t_col_inds
        );
    }
};

}  // namespace openturbine
