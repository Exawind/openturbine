#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "compute_t_col_inds.hpp"
#include "compute_t_num_non_zero.hpp"
#include "compute_t_row_ptrs.hpp"

#include "src/dof_management/freedom_signature.hpp"

namespace openturbine {

/// Creates the tangent operator matrix T in sparse CRS format
template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateTMatrix(
    size_t system_dofs,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table
) {
    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    const auto T_num_rows = system_dofs;
    const auto T_num_columns = T_num_rows;
    const auto T_num_non_zero = ComputeTNumNonZero(node_freedom_allocation_table);

    const auto T_row_ptrs = ComputeTRowPtrs<RowPtrType>(
        T_num_rows, node_freedom_allocation_table, node_freedom_map_table
    );
    const auto T_col_inds = ComputeTColInds<RowPtrType, IndicesType>(
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
        T_col_inds};
}
}  // namespace openturbine
