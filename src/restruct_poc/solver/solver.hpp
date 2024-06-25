#pragma once

#include <array>
#include <vector>

#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>

#include "compute_number_of_non_zeros.hpp"
#include "constraint_input.hpp"
#include "constraints.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_indices_constraints.hpp"
#include "populate_sparse_indices_constraints_transpose.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "populate_sparse_row_ptrs_constraints.hpp"
#include "populate_sparse_row_ptrs_constraints_transpose.hpp"
#include "state.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Solver {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using CrsMatrixType = KokkosSparse::CrsMatrix<double, int, DeviceType>;
    using DenseMatrixType = Kokkos::View<double**, Kokkos::LayoutLeft>;
    using ValuesType = Kokkos::View<CrsMatrixType::value_type*>;
    using RowPtrType = Kokkos::View<CrsMatrixType::size_type*>;
    using IndicesType = Kokkos::View<CrsMatrixType::ordinal_type*>;
    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<
        unsigned, int, double, ExecutionSpace, MemorySpace, MemorySpace>;
    using SpmvHandle = KokkosSparse::SPMVHandle<
        ExecutionSpace, CrsMatrixType, Kokkos::View<double*>, Kokkos::View<double*>>;
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
    int num_constraint_nodes;
    int num_constraint_dofs;
    int num_dofs;
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
    Kokkos::View<int*, Kokkos::LayoutLeft> IPIV;
    View_N R;  // System residual vector
    View_N x;  // System solution vector
    State state;
    Constraints constraints;
    std::vector<double> convergence_err;

    Solver(
        bool is_dynamic_solve_, int max_iter_, double h_, double rho_inf, int num_system_nodes_,
        Beams& beams_,
        std::vector<ConstraintInput> constraint_inputs = std::vector<ConstraintInput>(),
        std::vector<std::array<double, 7>> q_ = std::vector<std::array<double, 7>>(),
        std::vector<std::array<double, 6>> v_ = std::vector<std::array<double, 6>>(),
        std::vector<std::array<double, 6>> vd_ = std::vector<std::array<double, 6>>()
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
          num_system_nodes(num_system_nodes_),
          num_system_dofs(num_system_nodes * kLieAlgebraComponents),
          num_constraint_nodes(constraint_inputs.size()),
          num_constraint_dofs(num_constraint_nodes * kLieAlgebraComponents),
          num_dofs(num_system_dofs + num_constraint_dofs),
          K_dense("K dense", num_system_dofs, num_system_dofs),
          IPIV("IPIV", num_dofs),
          R("R", num_dofs),
          x("x", num_dofs),
          state(num_system_nodes, num_constraint_nodes, q_, v_, vd_),
          constraints(constraint_inputs, num_system_nodes),
          convergence_err(max_iter) {

        auto num_rows = num_system_dofs;
        auto num_columns = num_system_dofs;
        auto num_non_zero = 0;
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros", beams_.num_elems,
            ComputeNumberOfNonZeros{beams_.elem_indices}, num_non_zero
        );
        auto row_ptrs = RowPtrType("row_ptrs", num_rows + 1);
        auto indices = IndicesType("indices", num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrs", 1, PopulateSparseRowPtrs<CrsMatrixType::size_type>{beams_.elem_indices, row_ptrs}
        );
        Kokkos::parallel_for(
            "PopulateSparseIndices", 1,
            PopulateSparseIndices{beams_.elem_indices, beams_.node_state_indices, indices}
        );

        Kokkos::fence();
        auto K_values = ValuesType("K values", num_non_zero);
        K = CrsMatrixType("K", num_rows, num_columns, num_non_zero, K_values, row_ptrs, indices);
        auto T_values = ValuesType("T values", num_non_zero);
        T = CrsMatrixType("T", num_rows, num_columns, num_non_zero, T_values, row_ptrs, indices);

        auto B_num_rows = num_constraint_dofs;
        auto B_num_columns = num_system_dofs;
        auto B_num_non_zero = num_constraint_nodes * 6 * 6;
        auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
        auto B_indices = IndicesType("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrs_Constraints", 1,
            PopulateSparseRowPtrs_Constraints<CrsMatrixType::size_type>{num_constraint_nodes, B_row_ptrs}
        );

        Kokkos::parallel_for(
            "PopulateSparseIndices_Constraints", 1,
            PopulateSparseIndices_Constraints{
                num_constraint_nodes, constraints.node_indices, B_indices}
        );

        auto B_values = ValuesType("B values", B_num_non_zero);
        B = CrsMatrixType(
            "B", B_num_rows, B_num_columns, B_num_non_zero, B_values, B_row_ptrs, B_indices
        );

        auto B_t_num_rows = B_num_columns;
        auto B_t_num_columns = B_num_rows;
        auto B_t_num_non_zero = B_num_non_zero;
        auto B_t_row_ptrs = RowPtrType("b_t_row_ptrs", B_t_num_rows + 1);
        auto B_t_indices = IndicesType("B_t_indices", B_t_num_non_zero);

        Kokkos::parallel_for(
            "PopulateSparseRowPtrs_Constraints_Transpose", 1,
            PopulateSparseRowPtrs_Constraints_Transpose<CrsMatrixType::size_type>{
                num_constraint_nodes, num_system_nodes, constraints.node_indices, B_t_row_ptrs}
        );
        Kokkos::parallel_for(
            "PopulateSparseIndices_Constraints_Transpose", 1,
            PopulateSparseIndices_Constraints_Transpose{
                num_constraint_nodes, constraints.node_indices, B_t_indices}
        );

        auto B_t_values = ValuesType("B_t values", B_t_num_non_zero);
        B_t = CrsMatrixType(
            "B_t", B_t_num_rows, B_t_num_columns, B_t_num_non_zero, B_t_values, B_t_row_ptrs,
            B_t_indices
        );
    }
};

}  // namespace openturbine
