#pragma once

#include <array>
#include <vector>

#include <Amesos2.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>

#include "compute_number_of_non_zeros.hpp"
#include "compute_number_of_non_zeros_constraints.hpp"
#include "fill_unshifted_row_ptrs.hpp"
#include "populate_sparse_indices.hpp"
#include "populate_sparse_row_ptrs.hpp"
#include "populate_sparse_row_ptrs_col_inds_constraints.hpp"
#include "populate_sparse_row_ptrs_col_inds_transpose.hpp"
#include "populate_tangent_indices.hpp"
#include "populate_tangent_row_ptrs.hpp"

#include "src/constraints/constraint_type.hpp"

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

    size_t num_system_nodes;  //< Number of system nodes
    size_t num_system_dofs;   //< Number of system degrees of freedom
    size_t num_dofs;          //< Number of degrees of freedom

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
    Kokkos::View<double*> R;                   // System residual vector
    Kokkos::View<double*> x;                   // System solution vector

    std::vector<double> convergence_err;

    KernelHandle system_spgemm_handle;
    KernelHandle constraints_spgemm_handle;
    KernelHandle system_spadd_handle;
    KernelHandle spc_spadd_handle;
    KernelHandle full_system_spadd_handle;

    Teuchos::RCP<Amesos2::Solver<GlobalCrsMatrixType, GlobalMultiVectorType>> amesos_solver;

    Solver(
        const Kokkos::View<size_t*>::const_type& node_IDs,
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices, size_t num_constraint_dofs,
        const Kokkos::View<ConstraintType*>::const_type& constraint_type,
        const Kokkos::View<size_t*>::const_type& constraint_base_node_index,
        const Kokkos::View<size_t*>::const_type& constraint_target_node_index,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
    )
        : num_system_nodes(node_IDs.extent(0)),
          num_system_dofs(num_system_nodes * kLieAlgebraComponents),
          num_dofs(num_system_dofs + num_constraint_dofs),
          R("R", num_dofs),
          x("x", num_dofs) {
        auto K_num_rows = this->num_system_dofs;
        auto K_num_columns = K_num_rows;
        auto K_num_non_zero = size_t{0U};
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros", num_nodes_per_element.extent(0),
            ComputeNumberOfNonZeros{num_nodes_per_element}, K_num_non_zero
        );
        auto K_row_ptrs = RowPtrType("K_row_ptrs", K_num_rows + 1);
        auto K_col_inds = IndicesType("indices", K_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrs", 1,
            PopulateSparseRowPtrs<RowPtrType>{num_nodes_per_element, K_row_ptrs}
        );
        Kokkos::parallel_for(
            "PopulateSparseIndices", 1,
            PopulateSparseIndices{num_nodes_per_element, node_state_indices, K_col_inds}
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

        Kokkos::parallel_for(
            "PopulateTangentIndices", 1,
            PopulateTangentIndices{this->num_system_nodes, node_IDs, T_indices}
        );
        auto T_values = ValuesType("T values", T_num_non_zero);
        T = CrsMatrixType(
            "T", static_cast<int>(K_num_rows), static_cast<int>(K_num_columns), T_num_non_zero,
            T_values, T_row_ptrs, T_indices
        );

        // Initialize contraint for indexing for sparse matrices
        auto B_num_non_zero = size_t{0U};
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros_Constraints", constraint_type.extent(0),
            ComputeNumberOfNonZeros_Constraints{constraint_type, constraint_row_range},
            B_num_non_zero
        );
        auto B_num_rows = num_constraint_dofs;
        auto B_num_columns = this->num_system_dofs;
        auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
        auto B_col_ind = IndicesType("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrsColInds_Constraints", 1,
            PopulateSparseRowPtrsColInds_Constraints<RowPtrType, IndicesType>{
                constraint_type, constraint_base_node_index, constraint_target_node_index,
                constraint_row_range, B_row_ptrs, B_col_ind
            }
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
                B_t_row_ptrs, B_t_col_inds
            }
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
                num_system_dofs, system_matrix.graph.row_map, system_matrix_full_row_ptrs
            }
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
                num_system_dofs, B_t.graph.row_map, transpose_matrix_full_row_ptrs
            }
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
