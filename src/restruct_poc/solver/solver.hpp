#pragma once

#include <array>
#include <vector>

#include <Amesos2.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>

#include "compute_number_of_non_zeros.hpp"
#include "compute_number_of_non_zeros_constraints.hpp"
#include "constraint.hpp"
#include "constraints.hpp"
#include "fill_unshifted_row_ptrs.hpp"
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

/// @brief Solver struct holds all solver data
/// @details Solver struct holds all solver data and provides methods to update the
/// solver variables during the simulation
struct Solver {
    // Define some types for the solver to make the code more readable
    using GlobalCrsMatrixType = Tpetra::CrsMatrix<>;
    using ExecutionSpace = GlobalCrsMatrixType::execution_space;
    using MemorySpace = GlobalCrsMatrixType::memory_space;
    using GlobalMapType = GlobalCrsMatrixType::map_type;
    using GlobalMultiVectorType = Tpetra::MultiVector<>;
    using DualViewType = GlobalMultiVectorType::dual_view_type;
    using CrsMatrixType = GlobalCrsMatrixType::local_matrix_device_type;
    using ValuesType = CrsMatrixType::values_type::non_const_type;
    using RowPtrType = CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    using ScalarType = GlobalCrsMatrixType::scalar_type;
    using LocalOrdinalType = GlobalCrsMatrixType::local_ordinal_type;
    using GlobalOrdinalType = GlobalCrsMatrixType::global_ordinal_type;
    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<
        RowPtrType::value_type, IndicesType::value_type, ValuesType::value_type, ExecutionSpace,
        MemorySpace, MemorySpace>;
    using SpmvHandle =
        KokkosSparse::SPMVHandle<ExecutionSpace, CrsMatrixType, ValuesType, ValuesType>;

    bool is_dynamic_solve;    //< Flag to indicate if the solver is dynamic
    size_t max_iter;          //< Maximum number of iterations
    double h;                 //< Time step
    double alpha_m;           //< Alpha_m coefficient
    double alpha_f;           //< Alpha_f coefficient
    double gamma;             //< Gamma coefficient
    double beta;              //< Beta coefficient
    double gamma_prime;       //< Gamma prime coefficient
    double beta_prime;        //< Beta prime coefficient
    double conditioner;       //< Conditioner for the system matrix
    size_t num_system_nodes;  //< Number of system nodes
    size_t num_system_dofs;   //< Number of system degrees of freedom
    Constraints constraints;  //< Constraints
    size_t num_dofs;          //< Number of degrees of freedom

    State state;                            //< State
    CrsMatrixType K;                        //< Stiffness matrix
    CrsMatrixType T;                        //< Tangent operator
    CrsMatrixType B;                        //< Constraints matrix
    CrsMatrixType B_t;                      //< Transpose of constraints matrix
    CrsMatrixType static_system_matrix;     //< Static system matrix
    CrsMatrixType system_matrix;            //< System matrix
    CrsMatrixType constraints_matrix;       //< Constraints matrix
    CrsMatrixType system_matrix_full;       //< System matrix with constraints
    CrsMatrixType constraints_matrix_full;  //< Constraints matrix with system
    CrsMatrixType transpose_matrix_full;
    CrsMatrixType system_plus_constraints;     //< System matrix with constraints
    CrsMatrixType full_matrix;                 //< Full system matrix
    Teuchos::RCP<GlobalCrsMatrixType> A;       // System matrix
    Teuchos::RCP<GlobalMultiVectorType> b;     // System RHS
    Teuchos::RCP<GlobalMultiVectorType> x_mv;  // System solution
    View_Nx6x6 T_dense;                        // Dense tangent operator
    Kokkos::View<double***> matrix_terms;      // Matrix terms
    View_N R;                                  // System residual vector
    View_N x;                                  // System solution vector

    std::vector<double> convergence_err;

    KernelHandle system_spgemm_handle;
    KernelHandle constraints_spgemm_handle;
    KernelHandle system_spadd_handle;
    KernelHandle spc_spadd_handle;
    KernelHandle full_system_spadd_handle;

    Teuchos::RCP<Amesos2::Solver<GlobalCrsMatrixType, GlobalMultiVectorType>> amesos_solver;

    Solver(
        bool is_dynamic_solve_, size_t max_iter_, double h_, double rho_inf,
        const std::vector<std::shared_ptr<Node>>& system_nodes,
        const std::vector<std::shared_ptr<Constraint>>& constraints_, Beams& beams_
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
          constraints(constraints_),
          num_dofs(num_system_dofs + constraints.num_dofs),
          state(num_system_nodes, constraints.num_dofs, system_nodes),
          T_dense("T dense", num_system_nodes),
          matrix_terms(
              "matrix_terms", beams_.num_elems, beams_.max_elem_nodes * kLieAlgebraComponents,
              beams_.max_elem_nodes * kLieAlgebraComponents
          ),
          R("R", num_dofs),
          x("x", num_dofs),
          convergence_err(max_iter) {
        auto K_num_rows = this->num_system_dofs;
        auto K_num_columns = this->num_system_dofs;
        auto K_num_non_zero = size_t{0U};
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros", beams_.num_elems,
            ComputeNumberOfNonZeros{beams_.elem_indices}, K_num_non_zero
        );
        auto K_row_ptrs = RowPtrType("K_row_ptrs", K_num_rows + 1);
        auto K_col_inds = IndicesType("indices", K_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrs", 1,
            PopulateSparseRowPtrs<RowPtrType>{beams_.elem_indices, K_row_ptrs}
        );
        Kokkos::parallel_for(
            "PopulateSparseIndices", 1,
            PopulateSparseIndices{beams_.elem_indices, beams_.node_state_indices, K_col_inds}
        );

        Kokkos::fence();
        auto K_values = ValuesType("K values", K_num_non_zero);
        K = CrsMatrixType(
            "K", static_cast<int>(K_num_rows), static_cast<int>(K_num_columns), K_num_non_zero,
            K_values, K_row_ptrs, K_col_inds
        );

        // Tangent operator sparse matrix
        auto T_num_non_zero = this->num_system_nodes * 6 * 6;
        auto T_row_ptrs = RowPtrType("T_row_ptrs", K_num_rows + 1);
        auto T_indices = IndicesType("T_indices", T_num_non_zero);
        Kokkos::parallel_for(
            "PopulateTangentRowPtrs", 1,
            PopulateTangentRowPtrs<CrsMatrixType::size_type>{this->num_system_nodes, T_row_ptrs}
        );
        auto node_ids = IndicesType("node_ids", system_nodes.size());
        auto host_node_ids = Kokkos::create_mirror(node_ids);

        for (auto i = 0U; i < system_nodes.size(); ++i) {
            host_node_ids(i) = static_cast<int>(system_nodes[i]->ID);
        }
        Kokkos::deep_copy(node_ids, host_node_ids);
        Kokkos::parallel_for(
            "PopulateTangentIndices", 1,
            PopulateTangentIndices{this->num_system_nodes, node_ids, T_indices}
        );
        auto T_values = ValuesType("T values", T_num_non_zero);
        T = CrsMatrixType(
            "T", static_cast<int>(K_num_rows), static_cast<int>(K_num_columns), T_num_non_zero,
            T_values, T_row_ptrs, T_indices
        );

        // Initialize contraint for indexing for sparse matrices
        auto B_num_non_zero = size_t{0U};
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros_Constraints", this->constraints.num,
            ComputeNumberOfNonZeros_Constraints{this->constraints.data}, B_num_non_zero
        );
        auto B_num_rows = this->constraints.num_dofs;
        auto B_num_columns = this->num_system_dofs;
        auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
        auto B_col_ind = IndicesType("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrsColInds_Constraints", 1,
            PopulateSparseRowPtrsColInds_Constraints<RowPtrType, IndicesType>{
                this->constraints.data, B_row_ptrs, B_col_ind}
        );
        auto B_values = ValuesType("B values", B_num_non_zero);
        KokkosSparse::sort_crs_matrix(B_row_ptrs, B_col_ind, B_values);
        B = CrsMatrixType(
            "B", static_cast<int>(B_num_rows), static_cast<int>(B_num_columns), B_num_non_zero,
            B_values, B_row_ptrs, B_col_ind
        );

        auto B_t_num_rows = B_num_columns;
        auto B_t_num_columns = B_num_rows;
        auto B_t_num_non_zero = B_num_non_zero;
        auto col_count = IndicesType("col_count", B_num_columns);
        auto tmp_row_ptrs = RowPtrType("tmp_row_ptrs", B_t_num_rows + 1);
        auto B_t_row_ptrs = RowPtrType("b_t_row_ptrs", B_t_num_rows + 1);
        auto B_t_col_inds = IndicesType("B_t_indices", B_t_num_non_zero);
        auto B_t_values = ValuesType("B_t values", B_t_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrsColInds_Transpose", 1,
            PopulateSparseRowPtrsColInds_Transpose<RowPtrType, IndicesType>{
                B_num_rows, B_num_columns, B_row_ptrs, B_col_ind, col_count, tmp_row_ptrs,
                B_t_row_ptrs, B_t_col_inds}
        );
        B_t = CrsMatrixType(
            "B_t", static_cast<int>(B_t_num_rows), static_cast<int>(B_t_num_columns),
            B_t_num_non_zero, B_t_values, B_t_row_ptrs, B_t_col_inds
        );

        system_spgemm_handle.create_spgemm_handle();
        KokkosSparse::spgemm_symbolic(
            system_spgemm_handle, K, false, T, false, static_system_matrix
        );
        KokkosSparse::spgemm_numeric(system_spgemm_handle, K, false, T, false, static_system_matrix);

        constraints_spgemm_handle.create_spgemm_handle();
        KokkosSparse::spgemm_symbolic(
            constraints_spgemm_handle, B, false, T, false, constraints_matrix
        );
        KokkosSparse::spgemm_numeric(
            constraints_spgemm_handle, B, false, T, false, constraints_matrix
        );

        system_spadd_handle.create_spadd_handle(true, true);
        KokkosSparse::spadd_symbolic(&system_spadd_handle, K, static_system_matrix, system_matrix);
        KokkosSparse::spadd_numeric(
            &system_spadd_handle, 1., K, 1., static_system_matrix, system_matrix
        );

        auto system_matrix_full_row_ptrs = RowPtrType("system_matrix_full_row_ptrs", num_dofs + 1);
        Kokkos::parallel_for(
            "FillUnshiftedRowPtrs", num_dofs + 1,
            FillUnshiftedRowPtrs<RowPtrType>{
                num_system_dofs, system_matrix.graph.row_map, system_matrix_full_row_ptrs}
        );
        system_matrix_full = CrsMatrixType(
            "system_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
            system_matrix.nnz(), system_matrix.values, system_matrix_full_row_ptrs,
            system_matrix.graph.entries
        );

        auto constraints_matrix_full_row_ptrs =
            RowPtrType("constraints_matrix_full_row_ptrs", num_dofs + 1);
        Kokkos::deep_copy(
            Kokkos::subview(
                constraints_matrix_full_row_ptrs, Kokkos::pair(num_system_dofs, num_dofs + 1)
            ),
            constraints_matrix.graph.row_map
        );
        constraints_matrix_full = CrsMatrixType(
            "constraints_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
            constraints_matrix.nnz(), constraints_matrix.values, constraints_matrix_full_row_ptrs,
            constraints_matrix.graph.entries
        );

        auto transpose_matrix_full_row_ptrs =
            RowPtrType("transpose_matrix_full_row_ptrs", num_dofs + 1);
        Kokkos::parallel_for(
            "FillUnshiftedRowPtrs", num_dofs + 1,
            FillUnshiftedRowPtrs<RowPtrType>{
                num_system_dofs, B_t.graph.row_map, transpose_matrix_full_row_ptrs}
        );
        auto transpose_matrix_full_indices = IndicesType("transpose_matrix_full_indices", B_t.nnz());
        Kokkos::deep_copy(transpose_matrix_full_indices, static_cast<int>(num_system_dofs));
        KokkosBlas::axpy(1, B_t.graph.entries, transpose_matrix_full_indices);
        transpose_matrix_full = CrsMatrixType(
            "transpose_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
            B_t.nnz(), B_t.values, transpose_matrix_full_row_ptrs, transpose_matrix_full_indices
        );

        spc_spadd_handle.create_spadd_handle(true, true);
        KokkosSparse::spadd_symbolic(
            &spc_spadd_handle, system_matrix_full, constraints_matrix_full, system_plus_constraints
        );
        KokkosSparse::spadd_numeric(
            &spc_spadd_handle, 1., system_matrix_full, 1., constraints_matrix_full,
            system_plus_constraints
        );

        full_system_spadd_handle.create_spadd_handle(true, true);
        KokkosSparse::spadd_symbolic(
            &full_system_spadd_handle, system_plus_constraints, transpose_matrix_full, full_matrix
        );
        KokkosSparse::spadd_numeric(
            &full_system_spadd_handle, 1., system_plus_constraints, 1., transpose_matrix_full,
            full_matrix
        );

        auto comm = Teuchos::createSerialComm<LocalOrdinalType>();
        auto rowMap = Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
            static_cast<size_t>(full_matrix.numRows()), comm
        );
        auto colMap = Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
            static_cast<size_t>(full_matrix.numCols()), comm
        );

        A = Teuchos::make_rcp<GlobalCrsMatrixType>(rowMap, colMap, CrsMatrixType("A", full_matrix));
        b = Tpetra::createMultiVector<ScalarType>(A->getRangeMap(), 1);
        x_mv = Tpetra::createMultiVector<ScalarType>(A->getDomainMap(), 1);

        const auto solver_name = (std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>)
                                     ? std::string{"klu2"}
                                     : std::string{"basker"};

        amesos_solver =
            Amesos2::create<GlobalCrsMatrixType, GlobalMultiVectorType>(solver_name, A, x_mv, b);
        amesos_solver->symbolicFactorization();
    }
};

}  // namespace openturbine
