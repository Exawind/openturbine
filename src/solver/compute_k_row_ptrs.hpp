#pragma once

#include <Kokkos_Core.hpp>

#include "src/dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename RowPtrType>
struct ComputeKRowEntries {
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    RowPtrType K_row_entries;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto this_node_num_dof = count_active_dofs(node_freedom_allocation_table(i));
        const auto this_node_dof_index = node_freedom_map_table(i);

        auto num_entries_in_row = this_node_num_dof;
        bool node_found_in_system = false;

        // contributions to non-diagonal block from coupled nodes
        for (auto e = 0U; e < num_nodes_per_element.extent(0); ++e) {
            bool contains_node = false;
            auto num_entries_in_element = 0UL;
            for (auto j = 0U; j < num_nodes_per_element(e); ++j) {
                contains_node = contains_node || (node_state_indices(e, j) == i);
                num_entries_in_element +=
                    count_active_dofs(node_freedom_allocation_table(node_state_indices(e, j)));
            }
            if (contains_node) {
                node_found_in_system = true;
                num_entries_in_row += num_entries_in_element - this_node_num_dof;
            }
        }
        if (node_found_in_system) {
            for (auto j = 0U; j < this_node_num_dof; ++j) {
                K_row_entries(this_node_dof_index + j) = num_entries_in_row;
            }
        }
    }
};

template <typename RowPtrType>
struct ComputeKRowPtrsScanner {
    typename RowPtrType::const_type K_row_entries;
    RowPtrType K_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update, bool is_final) const {
        update += K_row_entries(i);
        if (is_final) {
            K_row_ptrs(i + 1) = update;
        }
    }
};

/// Computes the row pointers for the sparse stiffness matrix K in CSR format
template <typename RowPtrType>
[[nodiscard]] inline RowPtrType ComputeKRowPtrs(
    size_t K_num_rows,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices
) {
    auto K_row_ptrs = RowPtrType("row_ptrs", K_num_rows + 1);
    const auto K_row_entries = RowPtrType("row_ptrs", K_num_rows);

    Kokkos::parallel_for(
        "ComputeKRowEntries", node_freedom_allocation_table.extent(0),
        ComputeKRowEntries<RowPtrType>{
            node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
            node_state_indices, K_row_entries
        }
    );

    auto result = 0UL;
    Kokkos::parallel_scan(
        "ComputeKRowPtrs", K_row_entries.extent(0),
        ComputeKRowPtrsScanner<RowPtrType>{K_row_entries, K_row_ptrs}, result
    );

    return K_row_ptrs;
}

}  // namespace openturbine
