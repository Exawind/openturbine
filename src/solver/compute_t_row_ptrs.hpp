#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename RowPtrType>
struct ComputeTRowEntries {
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    RowPtrType T_row_entries;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto this_node_num_dof = count_active_dofs(node_freedom_allocation_table(i));
        const auto this_node_dof_index = node_freedom_map_table(i);

        auto num_entries_in_row = this_node_num_dof;

        for (auto j = 0U; j < this_node_num_dof; ++j) {
            T_row_entries(this_node_dof_index + j) = num_entries_in_row;
        }
    }
};

template <typename RowPtrType>
struct ComputeTRowPtrsScanner {
    typename RowPtrType::const_type T_row_entries;
    RowPtrType T_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update, bool is_final) const {
        update += T_row_entries(i);
        if (is_final) {
            T_row_ptrs(i + 1) = update;
        }
    }
};

template <typename RowPtrType>
[[nodiscard]] inline RowPtrType ComputeTRowPtrs(
    size_t T_num_rows,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table
) {
    auto T_row_ptrs = RowPtrType("T_row_ptrs", T_num_rows + 1);
    const auto T_row_entries = RowPtrType("row_entries", T_num_rows);

    Kokkos::parallel_for(
        "ComputeTRowEntries", node_freedom_allocation_table.extent(0),
        ComputeTRowEntries<RowPtrType>{
            node_freedom_allocation_table, node_freedom_map_table, T_row_entries
        }
    );

    auto result = 0UL;
    Kokkos::parallel_scan(
        "ComputeTRowPtrs", T_row_entries.extent(0),
        ComputeTRowPtrsScanner<RowPtrType>{T_row_entries, T_row_ptrs}, result
    );

    return T_row_ptrs;
}
}  // namespace openturbine
