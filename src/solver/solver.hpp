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

/** @brief A linear systems solver for assembling and solving system matrices and constraints in
 * OpenTurbine. This wraps Trilinos packages (Tpetra, Amesos2) for handling sparse matrix
 * operations and linear systems solution.
 *
 * @details This solver manages the assembly and solution of linear systems arising from the
 * generalized-alpha based time integration of the dynamic structural problem. The linear systems
 * include:
 *   - System stiffness matrix (K)
 *   - Tangent operator matrix (T)
 *   - Constraint matrices (B and B_t)
 *   - Combined system matrices for the complete structural problem
 */
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

    CrsMatrixType K;                           //< Stiffness matrix
    CrsMatrixType T;                           //< Tangent operator
    CrsMatrixType B;                           //< Constraints matrix
    CrsMatrixType B_t;                         //< Transpose of constraints matrix
    CrsMatrixType static_system_matrix;        //< Static system matrix
    CrsMatrixType system_matrix;               //< System matrix
    CrsMatrixType constraints_matrix;          //< Constraints matrix
    CrsMatrixType system_matrix_full;          //< System matrix with constraints
    CrsMatrixType constraints_matrix_full;     //< Constraints matrix with system
    CrsMatrixType transpose_matrix_full;       //< Transpose of system matrix with constraints
    CrsMatrixType system_plus_constraints;     //< System matrix with constraints
    CrsMatrixType full_matrix;                 //< Full system matrix
    Teuchos::RCP<GlobalCrsMatrixType> A;       //< System matrix
    Teuchos::RCP<GlobalMultiVectorType> b;     //< System RHS
    Teuchos::RCP<GlobalMultiVectorType> x_mv;  //< System solution
    Kokkos::View<double*> R;                   //< System residual vector
    Kokkos::View<double*> x;                   //< System solution vector

    std::vector<double> convergence_err;

    KernelHandle system_spgemm_handle;
    KernelHandle constraints_spgemm_handle;
    KernelHandle system_spadd_handle;
    KernelHandle spc_spadd_handle;
    KernelHandle full_system_spadd_handle;

    Teuchos::RCP<Amesos2::Solver<GlobalCrsMatrixType, GlobalMultiVectorType>> amesos_solver;

private:
    /// Computes the total number of active degrees of freedom in the system
    [[nodiscard]] static size_t ComputeNumSystemDofs(
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table
    ) {
        const auto nfat_host = Kokkos::create_mirror(node_freedom_allocation_table);
        Kokkos::deep_copy(nfat_host, node_freedom_allocation_table);
        auto total_system_dofs = 0UL;
        for (auto i = 0U; i < nfat_host.extent(0); ++i) {
            total_system_dofs += count_active_dofs(nfat_host(i));
        }
        return total_system_dofs;
    }

    /// Computes the number of non-zero entries in the stiffness matrix K for sparse storage
    [[nodiscard]] static size_t ComputeKNumNonZero(
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table
    ) {
        const auto nfat_host = Kokkos::create_mirror(node_freedom_allocation_table);
        Kokkos::deep_copy(nfat_host, node_freedom_allocation_table);
        const auto nnpe_host = Kokkos::create_mirror(num_nodes_per_element);
        Kokkos::deep_copy(nnpe_host, num_nodes_per_element);
        const auto nsi_host = Kokkos::create_mirror(node_state_indices);
        Kokkos::deep_copy(nsi_host, node_state_indices);

        auto K_num_non_zero = 0UL;
        // Contributions to non-diagonal blocks from coupled nodes
        for (auto i = 0U; i < nnpe_host.extent(0); ++i) {
            auto num_element_dof = 0UL;
            for (auto j = 0U; j < nnpe_host(i); ++j) {
                const auto num_node_dof = count_active_dofs(nfat_host(nsi_host(i, j)));
                num_element_dof += num_node_dof;
            }
            const auto num_element_non_zero = num_element_dof * num_element_dof;
            auto num_diagonal_non_zero = 0UL;
            for (auto j = 0U; j < nnpe_host(i); ++j) {
                const auto num_node_dof = count_active_dofs(nfat_host(nsi_host(i, j)));
                num_diagonal_non_zero += num_node_dof * num_node_dof;
            }
            K_num_non_zero += num_element_non_zero - num_diagonal_non_zero;
        }
        // Contributions to diagonal blocks for each node
        for (auto i = 0U; i < nfat_host.extent(0); ++i) {
            const auto num_node_dof = count_active_dofs(nfat_host(i));
            const auto num_diagonal_non_zero = num_node_dof * num_node_dof;
            K_num_non_zero += num_diagonal_non_zero;
        }

        return K_num_non_zero;
    }

    /// Computes the row pointers for the sparse stiffness matrix K in CSR format
    [[nodiscard]] static RowPtrType ComputeKRowPtrs(
        size_t K_num_rows,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices
    ) {
        const auto nfat_host = Kokkos::create_mirror(node_freedom_allocation_table);
        Kokkos::deep_copy(nfat_host, node_freedom_allocation_table);

        const auto nfmt_host = Kokkos::create_mirror(node_freedom_map_table);
        Kokkos::deep_copy(nfmt_host, node_freedom_map_table);

        const auto nnpe_host = Kokkos::create_mirror(num_nodes_per_element);
        Kokkos::deep_copy(nnpe_host, num_nodes_per_element);

        const auto nsi_host = Kokkos::create_mirror(node_state_indices);
        Kokkos::deep_copy(nsi_host, node_state_indices);

        const auto K_row_ptrs = RowPtrType("row_ptrs", K_num_rows + 1);
        const auto K_row_ptrs_host = Kokkos::create_mirror(K_row_ptrs);

        const auto K_row_entries = RowPtrType("row_ptrs", K_num_rows);
        const auto K_row_entries_host = Kokkos::create_mirror(K_row_entries);

        for (auto i = 0U; i < nfat_host.extent(0); ++i) {
            const auto this_node_num_dof = count_active_dofs(nfat_host(i));
            const auto this_node_dof_index = nfmt_host(i);

            auto num_entries_in_row = this_node_num_dof;
            bool node_found_in_system = false;

            // contributions to non-diagonal block from coupled nodes
            for (auto e = 0U; e < nnpe_host.extent(0); ++e) {
                bool contains_node = false;
                auto num_entries_in_element = 0UL;
                for (auto j = 0U; j < nnpe_host(e); ++j) {
                    contains_node = contains_node || (nsi_host(e, j) == i);
                    num_entries_in_element += count_active_dofs(nfat_host(nsi_host(e, j)));
                }
                if (contains_node) {
                    node_found_in_system = true;
                    num_entries_in_row += num_entries_in_element - this_node_num_dof;
                }
            }
            if (node_found_in_system) {
                for (auto j = 0U; j < this_node_num_dof; ++j) {
                    K_row_entries_host(this_node_dof_index + j) = num_entries_in_row;
                }
            }
        }

        for (auto i = 0U; i < K_row_entries_host.extent(0); ++i) {
            K_row_ptrs_host(i + 1) = K_row_ptrs_host(i) + K_row_entries_host(i);
        }

        Kokkos::deep_copy(K_row_ptrs, K_row_ptrs_host);

        return K_row_ptrs;
    }

    // Suppress cognitive complexity check for now - this will be extended and refactored soon
    // NOLINTBEGIN(readability-function-cognitive-complexity)
    [[nodiscard]] static IndicesType ComputeKColInds(
        size_t K_num_non_zero,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices, const RowPtrType& K_row_ptrs
    ) {
        const auto nfat_host = Kokkos::create_mirror(node_freedom_allocation_table);
        Kokkos::deep_copy(nfat_host, node_freedom_allocation_table);

        const auto nfmt_host = Kokkos::create_mirror(node_freedom_map_table);
        Kokkos::deep_copy(nfmt_host, node_freedom_map_table);

        const auto nnpe_host = Kokkos::create_mirror(num_nodes_per_element);
        Kokkos::deep_copy(nnpe_host, num_nodes_per_element);

        const auto nsi_host = Kokkos::create_mirror(node_state_indices);
        Kokkos::deep_copy(nsi_host, node_state_indices);

        const auto K_row_ptrs_host = Kokkos::create_mirror(K_row_ptrs);
        Kokkos::deep_copy(K_row_ptrs_host, K_row_ptrs);

        auto K_col_inds = IndicesType("col_inds", K_num_non_zero);
        const auto K_col_inds_host = Kokkos::create_mirror(K_col_inds);

        for (auto i = 0U; i < nfat_host.extent(0); ++i) {
            const auto this_node_num_dof = count_active_dofs(nfat_host(i));
            const auto this_node_dof_index = nfmt_host(i);

            for (auto j = 0U; j < this_node_num_dof; ++j) {
                auto current_dof_index = K_row_ptrs_host(this_node_dof_index + j);

                for (auto k = 0U; k < this_node_num_dof; ++k, ++current_dof_index) {
                    K_col_inds_host(current_dof_index) = static_cast<int>(this_node_dof_index + k);
                }

                for (auto e = 0U; e < nnpe_host.extent(0); ++e) {
                    bool contains_node = false;
                    for (auto n = 0U; n < nnpe_host(e); ++n) {
                        contains_node = contains_node || (nsi_host(e, n) == i);
                    }
                    if (contains_node) {
                        for (auto n = 0U; n < nnpe_host(e); ++n) {
                            if (nsi_host(e, n) != i) {
                                const auto target_node_num_dof =
                                    count_active_dofs(nfat_host(nsi_host(e, n)));
                                const auto target_node_dof_index = nfmt_host(nsi_host(e, n));
                                for (auto k = 0U; k < target_node_num_dof;
                                     ++k, ++current_dof_index) {
                                    K_col_inds_host(current_dof_index) =
                                        static_cast<int>(target_node_dof_index + k);
                                }
                            }
                        }
                    }
                }
            }
        }
        Kokkos::deep_copy(K_col_inds, K_col_inds_host);
        return K_col_inds;
    }
    // NOLINTEND(readability-function-cognitive-complexity)

    [[nodiscard]] static size_t ComputeTNumNonZero(
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table
    ) {
        const auto nfat_host = Kokkos::create_mirror(node_freedom_allocation_table);
        Kokkos::deep_copy(nfat_host, node_freedom_allocation_table);

        auto T_num_non_zero = 0UL;
        for (auto i = 0U; i < nfat_host.extent(0); ++i) {
            const auto num_node_dof = count_active_dofs(nfat_host(i));
            const auto num_diagonal_non_zero = num_node_dof * num_node_dof;
            T_num_non_zero += num_diagonal_non_zero;
        }

        return T_num_non_zero;
    }

    [[nodiscard]] static RowPtrType ComputeTRowPtrs(
        size_t T_num_rows,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table
    ) {
        const auto nfat_host = Kokkos::create_mirror(node_freedom_allocation_table);
        Kokkos::deep_copy(nfat_host, node_freedom_allocation_table);

        const auto nfmt_host = Kokkos::create_mirror(node_freedom_map_table);
        Kokkos::deep_copy(nfmt_host, node_freedom_map_table);

        const auto T_row_ptrs = RowPtrType("T_row_ptrs", T_num_rows + 1);
        const auto T_row_ptrs_host = Kokkos::create_mirror(T_row_ptrs);

        const auto T_row_entries = RowPtrType("row_entries", T_num_rows);
        const auto T_row_entries_host = Kokkos::create_mirror(T_row_entries);

        for (auto i = 0U; i < nfat_host.extent(0); ++i) {
            const auto this_node_num_dof = count_active_dofs(nfat_host(i));
            const auto this_node_dof_index = nfmt_host(i);

            auto num_entries_in_row = this_node_num_dof;

            for (auto j = 0U; j < this_node_num_dof; ++j) {
                T_row_entries_host(this_node_dof_index + j) = num_entries_in_row;
            }
        }

        for (auto i = 0U; i < T_row_entries_host.extent(0); ++i) {
            T_row_ptrs_host(i + 1) = T_row_ptrs_host(i) + T_row_entries_host(i);
        }

        Kokkos::deep_copy(T_row_ptrs, T_row_ptrs_host);

        return T_row_ptrs;
    }

    [[nodiscard]] static IndicesType ComputeTColInds(
        size_t T_num_non_zero,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table, const RowPtrType& T_row_ptrs
    ) {
        const auto nfat_host = Kokkos::create_mirror(node_freedom_allocation_table);
        Kokkos::deep_copy(nfat_host, node_freedom_allocation_table);

        const auto nfmt_host = Kokkos::create_mirror(node_freedom_map_table);
        Kokkos::deep_copy(nfmt_host, node_freedom_map_table);

        const auto T_row_ptrs_host = Kokkos::create_mirror(T_row_ptrs);
        Kokkos::deep_copy(T_row_ptrs_host, T_row_ptrs);

        const auto T_col_inds = IndicesType("T_indices", T_num_non_zero);
        const auto T_col_inds_host = Kokkos::create_mirror(T_col_inds);

        for (auto i = 0U; i < nfat_host.extent(0); ++i) {
            const auto this_node_num_dof = count_active_dofs(nfat_host(i));
            const auto this_node_dof_index = nfmt_host(i);

            for (auto j = 0U; j < this_node_num_dof; ++j) {
                auto current_dof_index = T_row_ptrs_host(this_node_dof_index + j);

                for (auto k = 0U; k < this_node_num_dof; ++k, ++current_dof_index) {
                    T_col_inds_host(current_dof_index) = static_cast<int>(this_node_dof_index + k);
                }
            }
        }

        Kokkos::deep_copy(T_col_inds, T_col_inds_host);

        return T_col_inds;
    }

    [[nodiscard]] static size_t ComputeBNumNonZero(
        const Kokkos::View<ConstraintType*>::const_type& constraint_type,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
    ) {
        auto B_num_non_zero = size_t{0U};
        Kokkos::parallel_reduce(
            "ComputeNumberOfNonZeros_Constraints", constraint_type.extent(0),
            ComputeNumberOfNonZeros_Constraints{constraint_type, constraint_row_range},
            B_num_non_zero
        );
        return B_num_non_zero;
    }

    /// Creates the system stiffness matrix K in sparse CRS format
    [[nodiscard]] static CrsMatrixType CreateKMatrix(
        size_t system_dofs,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices
    ) {
        const auto K_num_rows = system_dofs;
        const auto K_num_columns = K_num_rows;
        const auto K_num_non_zero = ComputeKNumNonZero(
            num_nodes_per_element, node_state_indices, node_freedom_allocation_table
        );

        const auto K_row_ptrs = ComputeKRowPtrs(
            K_num_rows, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
            node_state_indices
        );
        const auto K_col_inds = ComputeKColInds(
            K_num_non_zero, node_freedom_allocation_table, node_freedom_map_table,
            num_nodes_per_element, node_state_indices, K_row_ptrs
        );
        const auto K_values = ValuesType("K values", K_num_non_zero);

        KokkosSparse::sort_crs_matrix(K_row_ptrs, K_col_inds, K_values);
        return {
            "K",
            static_cast<int>(K_num_rows),
            static_cast<int>(K_num_columns),
            K_num_non_zero,
            K_values,
            K_row_ptrs,
            K_col_inds
        };
    }

    /// Creates the tangent operator matrix T in sparse CRS format
    [[nodiscard]] static CrsMatrixType CreateTMatrix(
        size_t system_dofs,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table
    ) {
        const auto T_num_rows = system_dofs;
        const auto T_num_columns = T_num_rows;
        const auto T_num_non_zero = ComputeTNumNonZero(node_freedom_allocation_table);

        const auto T_row_ptrs =
            ComputeTRowPtrs(T_num_rows, node_freedom_allocation_table, node_freedom_map_table);
        const auto T_col_inds = ComputeTColInds(
            T_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, T_row_ptrs
        );
        const auto T_values = ValuesType("T values", T_num_non_zero);

        KokkosSparse::sort_crs_matrix(T_row_ptrs, T_col_inds, T_values);
        return {
            "T",
            static_cast<int>(T_num_rows),
            static_cast<int>(T_num_columns),
            T_num_non_zero,
            T_values,
            T_row_ptrs,
            T_col_inds
        };
    }

    /// Creates the constraint matrix B in sparse CRS format
    [[nodiscard]] static CrsMatrixType CreateBMatrix(
        size_t system_dofs, size_t constraint_dofs,
        const Kokkos::View<ConstraintType*>::const_type& constraint_type,
        const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
        const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
    ) {
        const auto B_num_rows = constraint_dofs;
        const auto B_num_columns = system_dofs;
        const auto B_num_non_zero = ComputeBNumNonZero(constraint_type, constraint_row_range);

        const auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
        const auto B_col_ind = IndicesType("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrsColInds_Constraints", 1,
            PopulateSparseRowPtrsColInds_Constraints<RowPtrType, IndicesType>{
                constraint_type, constraint_base_node_freedom_table,
                constraint_target_node_freedom_table, constraint_row_range, B_row_ptrs, B_col_ind
            }
        );
        const auto B_values = ValuesType("B values", B_num_non_zero);
        KokkosSparse::sort_crs_matrix(B_row_ptrs, B_col_ind, B_values);
        return {
            "B",
            static_cast<int>(B_num_rows),
            static_cast<int>(B_num_columns),
            B_num_non_zero,
            B_values,
            B_row_ptrs,
            B_col_ind
        };
    }

    /// Creates the constraint matrix B_t in sparse CRS format
    [[nodiscard]] static CrsMatrixType CreateBtMatrix(
        size_t system_dofs, size_t constraint_dofs,
        const Kokkos::View<ConstraintType*>::const_type& constraint_type,
        const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
        const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
    ) {
        const auto B_num_rows = constraint_dofs;
        const auto B_num_columns = system_dofs;
        const auto B_num_non_zero = ComputeBNumNonZero(constraint_type, constraint_row_range);

        const auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
        const auto B_col_ind = IndicesType("b_indices", B_num_non_zero);
        Kokkos::parallel_for(
            "PopulateSparseRowPtrsColInds_Constraints", 1,
            PopulateSparseRowPtrsColInds_Constraints<RowPtrType, IndicesType>{
                constraint_type, constraint_base_node_freedom_table,
                constraint_target_node_freedom_table, constraint_row_range, B_row_ptrs, B_col_ind
            }
        );
        const auto B_values = ValuesType("B values", B_num_non_zero);
        KokkosSparse::sort_crs_matrix(B_row_ptrs, B_col_ind, B_values);

        const auto B_t_num_rows = system_dofs;
        const auto B_t_num_columns = constraint_dofs;
        const auto B_t_num_non_zero = ComputeBNumNonZero(constraint_type, constraint_row_range);

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
        KokkosSparse::sort_crs_matrix(B_t_row_ptrs, B_t_col_inds, B_t_values);
        return {
            "B_t",
            static_cast<int>(B_t_num_rows),
            static_cast<int>(B_t_num_columns),
            B_t_num_non_zero,
            B_t_values,
            B_t_row_ptrs,
            B_t_col_inds
        };
    }

public:
    /** @brief Constructs a sparse linear systems solver for OpenTurbine
     *
     * @param node_IDs View containing the global IDs for each node
     * @param node_freedom_allocation_table View containing the freedom signature for each node
     * @param node_freedom_map_table View mapping node indices to DOF indices
     * @param num_nodes_per_element View containing number of nodes per element
     * @param node_state_indices View containing element-to-node connectivity
     * @param num_constraint_dofs Number of constraint degrees of freedom
     * @param constraint_type View containing the type of each constraint
     * @param constraint_base_node_freedom_table View containing base node DOFs for constraints
     * @param constraint_target_node_freedom_table View containing target node DOFs for constraints
     * @param constraint_row_range View containing row ranges for each constraint
     */
    Solver(
        const Kokkos::View<size_t*>::const_type& node_IDs,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices, size_t num_constraint_dofs,
        const Kokkos::View<ConstraintType*>::const_type& constraint_type,
        const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
        const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
    )
        : num_system_nodes(node_IDs.extent(0)),
          num_system_dofs(ComputeNumSystemDofs(node_freedom_allocation_table)),
          num_dofs(num_system_dofs + num_constraint_dofs),
          K(CreateKMatrix(
              num_system_dofs, node_freedom_allocation_table, node_freedom_map_table,
              num_nodes_per_element, node_state_indices
          )),
          T(CreateTMatrix(num_system_dofs, node_freedom_allocation_table, node_freedom_map_table)),
          B(CreateBMatrix(
              num_system_dofs, num_constraint_dofs, constraint_type,
              constraint_base_node_freedom_table, constraint_target_node_freedom_table,
              constraint_row_range
          )),
          B_t(CreateBtMatrix(
              num_system_dofs, num_constraint_dofs, constraint_type,
              constraint_base_node_freedom_table, constraint_target_node_freedom_table,
              constraint_row_range
          )),
          R("R", num_dofs),
          x("x", num_dofs) {
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
        auto row_map = Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
            static_cast<size_t>(full_matrix.numRows()), comm
        );
        auto col_map = Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
            static_cast<size_t>(full_matrix.numCols()), comm
        );

        A = Teuchos::make_rcp<GlobalCrsMatrixType>(
            row_map, col_map, CrsMatrixType("A", full_matrix)
        );
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
