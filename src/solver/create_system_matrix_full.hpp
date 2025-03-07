#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_k_col_inds.hpp"
#include "compute_k_num_non_zero.hpp"
#include "compute_k_row_ptrs.hpp"
#include "dof_management/freedom_signature.hpp"
#include "fill_unshifted_row_ptrs.hpp"

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateSystemMatrixFull(
    size_t num_system_dofs,
    size_t num_dofs,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full System Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::index_type::non_const_type;

    const auto K_num_rows = num_system_dofs;
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

    auto system_matrix_full_row_ptrs = RowPtrType("system_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::parallel_for(
        "FillUnshiftedRowPtrs", num_dofs + 1,
        FillUnshiftedRowPtrs<RowPtrType>{
            num_system_dofs, K_row_ptrs, system_matrix_full_row_ptrs
        }
    );
    return CrsMatrixType(
        "system_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
        K_num_non_zero, K_values, system_matrix_full_row_ptrs,
        K_col_inds
    );
}

}  // namespace openturbine
