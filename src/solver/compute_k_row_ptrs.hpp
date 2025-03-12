#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename RowPtrType>
struct ComputeKRowEntries {
    Kokkos::View<size_t*>::const_type active_dofs;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    RowPtrType K_row_entries;

    KOKKOS_FUNCTION
    bool ContainsNode(size_t element, size_t node) const {
        const auto num_nodes = num_nodes_per_element(element);
        for (auto n = 0U; n < num_nodes; ++n) {
            if(node_state_indices(element, n) == node) return true;
        }
        return false;
    }

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto this_node_num_dof = active_dofs(i);
        const auto this_node_dof_index = node_freedom_map_table(i);

        auto num_entries_in_row = this_node_num_dof;

        for (auto e = 0U; e < num_nodes_per_element.extent(0); ++e) {
            if (!ContainsNode(e, i)) {
                continue;
            }
            const auto num_nodes = num_nodes_per_element(e);
            auto num_entries_in_element = 0UL;
            for (auto j = 0U; j < num_nodes; ++j) {
                const auto node_state_index = node_state_indices(e, j);
                num_entries_in_element += active_dofs(node_state_index);
            }
            num_entries_in_row += num_entries_in_element - this_node_num_dof;

        }

        for (auto j = 0U; j < this_node_num_dof; ++j) {
            K_row_entries(this_node_dof_index + j) = num_entries_in_row;
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
    const Kokkos::View<size_t*>::const_type& active_dofs,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices
) {
    auto K_row_ptrs = RowPtrType("row_ptrs", K_num_rows + 1);
    const auto K_row_entries = RowPtrType("row_ptrs", K_num_rows);

    Kokkos::parallel_for(
        "ComputeKRowEntries", active_dofs.extent(0),
        ComputeKRowEntries<RowPtrType>{
            active_dofs, node_freedom_map_table, num_nodes_per_element,
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
