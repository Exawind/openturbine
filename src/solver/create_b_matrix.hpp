#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_b_num_non_zero.hpp"
#include "populate_sparse_row_ptrs_col_inds_constraints.hpp"

namespace openturbine {

/// Creates the constraint matrix B in sparse CRS format
template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateBMatrix(
    size_t system_dofs, size_t constraint_dofs,
    const Kokkos::View<ConstraintType*>::const_type& constraint_type,
    const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_base_node_col_range,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_target_node_col_range
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create B Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    const auto B_num_rows = constraint_dofs;
    const auto B_num_columns = system_dofs;
    const auto B_num_non_zero = ComputeBNumNonZero(
        constraint_type, constraint_row_range, constraint_base_node_col_range,
        constraint_target_node_col_range
    );

    const auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
    const auto B_col_ind = IndicesType("b_indices", B_num_non_zero);
    Kokkos::parallel_for(
        "PopulateSparseRowPtrsColInds_Constraints", 1,
        PopulateSparseRowPtrsColInds_Constraints<RowPtrType, IndicesType>{
            constraint_type, constraint_base_node_freedom_table,
            constraint_target_node_freedom_table, constraint_row_range,
            constraint_base_node_col_range, constraint_target_node_col_range, B_row_ptrs, B_col_ind
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
}  // namespace openturbine
