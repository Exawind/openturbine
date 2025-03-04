#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_k_col_inds.hpp"
#include "compute_k_num_non_zero.hpp"
#include "compute_k_row_ptrs.hpp"
#include "dof_management/freedom_signature.hpp"

namespace openturbine {

/// Creates the system stiffness matrix K in sparse CRS format
template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateKMatrix(
    size_t system_dofs,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create System Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::index_type::non_const_type;

    const auto K_num_rows = system_dofs;
    const auto K_num_columns = K_num_rows;
    const auto K_num_non_zero =
        ComputeKNumNonZero(num_nodes_per_element, node_state_indices, node_freedom_allocation_table);

    const auto K_row_ptrs = ComputeKRowPtrs<RowPtrType>(
        K_num_rows, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices
    );
    const auto K_col_inds = ComputeKColInds<RowPtrType, IndicesType>(
        K_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, K_row_ptrs
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
}  // namespace openturbine
