#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_b_num_non_zero.hpp"
#include "populate_sparse_row_ptrs_col_inds_constraints.hpp"
#include "populate_sparse_row_ptrs_col_inds_transpose.hpp"
#include "fill_unshifted_row_ptrs.hpp"

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateTransposeMatrixFull(
    size_t num_system_dofs,
    size_t num_dofs,
    size_t num_constraint_dofs,
    const Kokkos::View<ConstraintType*>::const_type& constraint_type,
    const Kokkos::View<FreedomSignature*>::const_type& base_node_freedom_signature,
    const Kokkos::View<FreedomSignature*>::const_type& target_node_freedom_signature,
    const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full Transpose Constraints Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    const auto B_num_rows = num_constraint_dofs;
    const auto B_num_columns = num_system_dofs;

    const auto B_num_non_zero = ComputeBNumNonZero(
        constraint_row_range, base_node_freedom_signature, target_node_freedom_signature
    );

    const auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
    const auto B_col_ind = IndicesType("b_indices", B_num_non_zero);
    Kokkos::parallel_for(
        "PopulateSparseRowPtrsColInds_Constraints", 1,
        PopulateSparseRowPtrsColInds_Constraints<RowPtrType, IndicesType>{
            constraint_type, constraint_base_node_freedom_table,
            constraint_target_node_freedom_table, constraint_row_range, base_node_freedom_signature,
            target_node_freedom_signature, B_row_ptrs, B_col_ind
        }
    );

    const auto B_values = ValuesType("B values", B_num_non_zero);
    KokkosSparse::sort_crs_matrix(B_row_ptrs, B_col_ind, B_values);

    const auto B_t_num_rows = num_system_dofs;
    const auto B_t_num_non_zero = B_num_non_zero;

    auto col_count = IndicesType("col_count", B_num_columns);
    auto tmp_row_ptrs = RowPtrType("tmp_row_ptrs", B_t_num_rows + 1);
    auto B_t_row_ptrs = RowPtrType("b_t_row_ptrs", B_t_num_rows + 1);
    auto B_t_col_inds = IndicesType("B_t_indices", B_t_num_non_zero);
    auto B_t_values = ValuesType("B_t values", B_t_num_non_zero);
    Kokkos::parallel_for(
        "PopulateSparseRowPtrsColInds_Transpose", 1,
        PopulateSparseRowPtrsColInds_Transpose<RowPtrType, IndicesType>{
            B_num_rows, B_num_columns, B_row_ptrs, B_col_ind, col_count, tmp_row_ptrs, B_t_row_ptrs,
            B_t_col_inds
        }
    );
    KokkosSparse::sort_crs_matrix(B_t_row_ptrs, B_t_col_inds, B_t_values);

    auto transpose_matrix_full_row_ptrs = RowPtrType("transpose_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::parallel_for(
        "FillUnshiftedRowPtrs", num_dofs + 1,
        FillUnshiftedRowPtrs<RowPtrType>{
            num_system_dofs, B_t_row_ptrs, transpose_matrix_full_row_ptrs
        }
    );
    auto transpose_matrix_full_indices = IndicesType("transpose_matrix_full_indices", B_t_num_non_zero);
    Kokkos::deep_copy(transpose_matrix_full_indices, static_cast<int>(num_system_dofs));
    KokkosBlas::axpy(1, B_t_col_inds, transpose_matrix_full_indices);
    return CrsMatrixType(
        "transpose_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs), B_t_num_non_zero,
        B_t_values, transpose_matrix_full_row_ptrs, transpose_matrix_full_indices
    );
}

}  // namespace openturbine
