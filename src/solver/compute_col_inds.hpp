#pragma once

#include <Kokkos_Core.hpp>

#include "compute_constraints_col_inds.hpp"
#include "compute_system_col_inds.hpp"

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
[[nodiscard]] inline IndicesType ComputeColInds(
    size_t num_non_zero,
    size_t num_system_dofs,
    const Kokkos::View<size_t*>::const_type& active_dofs,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices,
    const Kokkos::View<size_t*>::const_type& base_active_dofs,
    const Kokkos::View<size_t*>::const_type& target_active_dofs,
    const Kokkos::View<size_t* [6]>::const_type& base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& row_range,
    const typename RowPtrType::const_type& row_ptrs
) {
    const auto col_inds = IndicesType("col_inds", num_non_zero);

    const auto num_nodes = active_dofs.extent(0);
    const auto num_constraints = row_range.extent(0);

    Kokkos::parallel_for("ComputeSystemColInds", num_nodes, ComputeSystemColInds<RowPtrType, IndicesType>{num_system_dofs, active_dofs, node_freedom_map_table, num_nodes_per_element, node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table, target_node_freedom_table, row_range, row_ptrs, col_inds});

    Kokkos::parallel_for("ComputeConstraintsColInds", num_constraints, ComputeConstraintsColInds<RowPtrType, IndicesType>{num_system_dofs, base_active_dofs, target_active_dofs, base_node_freedom_table, target_node_freedom_table, row_range, row_ptrs, col_inds});

    return col_inds;
}
}
