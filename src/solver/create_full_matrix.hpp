#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_col_inds.hpp"
#include "compute_row_ptrs.hpp"

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateFullMatrix(
    size_t num_system_dofs, size_t num_dofs,
    const Kokkos::View<size_t*>::const_type& base_active_dofs,
    const Kokkos::View<size_t*>::const_type& target_active_dofs,
    const Kokkos::View<size_t* [6]>::const_type& base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& row_range,
    const Kokkos::View<size_t*>::const_type& active_dofs,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    using RowPtrValueType = typename RowPtrType::value_type;
    using IndicesValueType = typename IndicesType::value_type;

    const auto row_ptrs = ComputeRowPtrs<RowPtrType>(
        num_system_dofs, num_dofs, active_dofs, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table,
        target_node_freedom_table, row_range
    );

    auto num_non_zero = RowPtrValueType{};
    Kokkos::deep_copy(num_non_zero, Kokkos::subview(row_ptrs, num_dofs));

    const auto col_inds = ComputeColInds<RowPtrType, IndicesType>(
        num_non_zero, num_system_dofs, active_dofs, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table,
        target_node_freedom_table, row_range, row_ptrs
    );

    KokkosSparse::sort_crs_graph(row_ptrs, col_inds);

    return CrsMatrixType(
        "full_matrix", static_cast<IndicesValueType>(num_dofs),
        static_cast<IndicesValueType>(num_dofs), num_non_zero,
        ValuesType(
            Kokkos::view_alloc("values", Kokkos::WithoutInitializing),
            static_cast<size_t>(num_non_zero)
        ),
        row_ptrs, col_inds
    );
}

}  // namespace openturbine
