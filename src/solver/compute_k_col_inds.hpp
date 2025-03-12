#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct ComputeKColIndsFunction {
    using RowPtrValueType = typename RowPtrType::value_type;
    using IndicesValueType = typename IndicesType::value_type;
    Kokkos::View<size_t*>::const_type active_dofs;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    typename RowPtrType::const_type K_row_ptrs;
    IndicesType K_col_inds;

    KOKKOS_FUNCTION
    bool ContainsNode(size_t element, size_t node) const {
        const auto num_nodes = num_nodes_per_element(element);
        for (auto n = 0U; n < num_nodes; ++n) {
            if(node_state_indices(element, n) == node) return true;
        }
        return false;
    }

    KOKKOS_FUNCTION
    RowPtrValueType ComputeColInds(size_t element, size_t node, RowPtrValueType current_dof_index)
        const {
        for (auto n = 0U; n < num_nodes_per_element(element); ++n) {
            const auto node_state_index = node_state_indices(element, n);
            if (node_state_index != node) {
                const auto target_node_num_dof = active_dofs(node_state_index);
                const auto target_node_dof_index = node_freedom_map_table(node_state_index);
                for (auto k = 0U; k < target_node_num_dof; ++k, ++current_dof_index) {
                    const auto col_index = static_cast<IndicesValueType>(target_node_dof_index + k);
                    K_col_inds(current_dof_index) = col_index;
                }
            }
        }
        return current_dof_index;
    }

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto this_node_num_dof = active_dofs(i);
        const auto this_node_dof_index = node_freedom_map_table(i);
        auto current_col = Kokkos::Array<RowPtrValueType, 6>{};

        for (auto j = 0U; j < this_node_num_dof; ++j) {
            auto current_dof_index = K_row_ptrs(this_node_dof_index + j);

            for (auto k = 0U; k < this_node_num_dof; ++k, ++current_dof_index) {
                K_col_inds(current_dof_index) = static_cast<int>(this_node_dof_index + k);
            }
            current_col[j] = this_node_num_dof;
        }
        for (auto e = 0U; e < num_nodes_per_element.extent(0); ++e) {
            if (!ContainsNode(e, i)) {
                continue;
            }
            for (auto j = 0U; j < this_node_num_dof; ++j) {
                const auto current_dof_index = K_row_ptrs(this_node_dof_index + j) + current_col[j];
                const auto new_dof_index = ComputeColInds(e, i, current_dof_index);
                current_col[j] += new_dof_index - current_dof_index;
            }

        }

    }
};

template <typename RowPtrType, typename IndicesType>
[[nodiscard]] inline IndicesType ComputeKColInds(
    size_t K_num_non_zero,
    const Kokkos::View<size_t*>::const_type& active_dofs,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices,
    const typename RowPtrType::const_type& K_row_ptrs
) {
    auto K_col_inds = IndicesType("col_inds", K_num_non_zero);

    Kokkos::parallel_for(
        "ComputeKColInds", active_dofs.extent(0),
        ComputeKColIndsFunction<RowPtrType, IndicesType>{
            active_dofs, node_freedom_map_table, num_nodes_per_element,
            node_state_indices, K_row_ptrs, K_col_inds
        }
    );

    return K_col_inds;
}
}  // namespace openturbine
