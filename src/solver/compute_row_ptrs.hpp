#pragma once

#include <Kokkos_Core.hpp>

#include "compute_constraints_row_entries.hpp"
#include "compute_system_row_entries.hpp"
#include "scan_row_entries.hpp"

namespace openturbine {

template <typename RowPtrType>
[[nodiscard]] inline RowPtrType ComputeRowPtrs(
    size_t num_system_dofs,
    size_t num_dofs,
    const Kokkos::View<size_t*>::const_type& active_dofs,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices,
    const Kokkos::View<size_t*>::const_type& base_active_dofs,
    const Kokkos::View<size_t*>::const_type& target_active_dofs,
    const Kokkos::View<size_t* [6]>::const_type& base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& row_range
) {
    const auto row_entries = RowPtrType("row_entries", num_dofs);
    const auto row_ptrs = RowPtrType("row_ptrs", num_dofs + 1);

    const auto num_nodes = active_dofs.extent(0);
    const auto num_constraints = row_range.extent(0);

    Kokkos::parallel_for("ComputeSystemRowEntries", num_nodes, ComputeSystemRowEntries<RowPtrType>{active_dofs, node_freedom_map_table, num_nodes_per_element, node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table, target_node_freedom_table, row_range, row_entries});

    Kokkos::parallel_for("ComputeConstraintsRowEntries", num_constraints, ComputeConstraintsRowEntries<RowPtrType>{num_system_dofs, base_active_dofs, target_active_dofs, row_range, row_entries});

    Kokkos::parallel_scan("ScanRowEntries", num_dofs, ScanRowEntries<RowPtrType>{row_entries, row_ptrs});

    return row_ptrs;
}

}
