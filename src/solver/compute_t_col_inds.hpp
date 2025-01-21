#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct ComputeTColIndsFunction {
    using IndicesValuesType = typename IndicesType::value_type;
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    typename RowPtrType::const_type T_row_ptrs;
    IndicesType T_col_inds;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto this_node_num_dof = count_active_dofs(node_freedom_allocation_table(i));
        const auto this_node_dof_index = node_freedom_map_table(i);

        for (auto j = 0U; j < this_node_num_dof; ++j) {
            for (auto k = 0UL, current_dof_index = T_row_ptrs(this_node_dof_index + j);
                 k < this_node_num_dof; ++k, ++current_dof_index) {
                const auto col_index = static_cast<IndicesValuesType>(this_node_dof_index + k);
                T_col_inds(current_dof_index) = col_index;
            }
        }
    }
};

template <typename RowPtrType, typename IndicesType>
[[nodiscard]] inline IndicesType ComputeTColInds(
    size_t T_num_non_zero,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const typename RowPtrType::const_type& T_row_ptrs
) {
    auto T_col_inds = IndicesType("T_indices", T_num_non_zero);

    Kokkos::parallel_for(
        "ComputeTColInds", node_freedom_allocation_table.extent(0),
        ComputeTColIndsFunction<RowPtrType, IndicesType>{
            node_freedom_allocation_table, node_freedom_map_table, T_row_ptrs, T_col_inds
        }
    );

    return T_col_inds;
}

}  // namespace openturbine
