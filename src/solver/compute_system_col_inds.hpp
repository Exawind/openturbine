#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct ComputeSystemColInds {
    using RowPtrValueType = typename RowPtrType::value_type;
    using IndicesValueType = typename IndicesType::value_type;
    size_t num_system_dofs{};
    typename Kokkos::View<size_t*, typename RowPtrType::device_type>::const_type active_dofs;
    typename Kokkos::View<size_t*, typename RowPtrType::device_type>::const_type
        node_freedom_map_table;
    typename Kokkos::View<size_t*, typename RowPtrType::device_type>::const_type
        num_nodes_per_element;
    typename Kokkos::View<size_t**, typename RowPtrType::device_type>::const_type node_state_indices;
    typename Kokkos::View<size_t*, typename RowPtrType::device_type>::const_type base_active_dofs;
    typename Kokkos::View<size_t*, typename RowPtrType::device_type>::const_type target_active_dofs;
    typename Kokkos::View<size_t* [6], typename RowPtrType::device_type>::const_type
        base_node_freedom_table;
    typename Kokkos::View<size_t* [6], typename RowPtrType::device_type>::const_type
        target_node_freedom_table;
    typename Kokkos::View<
        Kokkos::pair<size_t, size_t>*, typename RowPtrType::device_type>::const_type row_range;
    typename RowPtrType::const_type row_ptrs;
    IndicesType col_inds;

    KOKKOS_FUNCTION
    bool ElementContainsNode(size_t element, size_t node) const {
        const auto num_nodes = num_nodes_per_element(element);
        for (auto node_idx = 0U; node_idx < num_nodes; ++node_idx) {
            if (node_state_indices(element, node_idx) == node) {
                return true;
            }
        }
        return false;
    }

    KOKKOS_FUNCTION
    bool BaseContainsNode(size_t constraint, size_t dof_index) const {
        return dof_index == base_node_freedom_table(constraint, 0) &&
               base_active_dofs(constraint) != 0UL;
    }

    KOKKOS_FUNCTION
    bool TargetContainsNode(size_t constraint, size_t dof_index) const {
        return dof_index == target_node_freedom_table(constraint, 0) &&
               target_active_dofs(constraint) != 0UL;
    }

    KOKKOS_FUNCTION
    bool ConstraintContainsNode(size_t constraint, size_t dof_index) const {
        return BaseContainsNode(constraint, dof_index) || TargetContainsNode(constraint, dof_index);
    }

    KOKKOS_FUNCTION
    RowPtrValueType ComputeColIndsInElement(size_t element, size_t node, RowPtrValueType index)
        const {
        for (auto node_index = 0U; node_index < num_nodes_per_element(element); ++node_index) {
            const auto node_state_index = node_state_indices(element, node_index);
            if (node_state_index != node) {
                const auto num_dof = active_dofs(node_state_index);
                const auto dof_index = node_freedom_map_table(node_state_index);
                for (auto component = 0U; component < num_dof; ++component, ++index) {
                    const auto col_index = dof_index + component;
                    col_inds(index) = static_cast<IndicesValueType>(col_index);
                }
            }
        }
        return index;
    }

    KOKKOS_FUNCTION
    RowPtrValueType ComputeColIndsInConstraint(size_t constraint, RowPtrValueType index) const {
        const auto first_row = row_range(constraint).first;
        const auto end_row = row_range(constraint).second;
        for (auto k = first_row; k < end_row; ++k, ++index) {
            const auto col_index = num_system_dofs + k;
            col_inds(index) = static_cast<IndicesValueType>(col_index);
        }
        return index;
    }

    KOKKOS_FUNCTION
    void operator()(size_t node) const {
        const auto num_dof = active_dofs(node);
        const auto dof_index = node_freedom_map_table(node);

        const auto num_elements = num_nodes_per_element.extent(0);
        const auto num_constraints = row_range.extent(0);

        auto current_col = Kokkos::Array<RowPtrValueType, 6>{};

        for (auto component_1 = 0U; component_1 < num_dof; ++component_1) {
            auto index = row_ptrs(dof_index + component_1);

            for (auto component_2 = 0U; component_2 < num_dof; ++component_2, ++index) {
                col_inds(index) = static_cast<IndicesValueType>(dof_index) +
                                  static_cast<IndicesValueType>(component_2);
            }
            current_col[component_1] = static_cast<RowPtrValueType>(num_dof);
        }

        for (auto element = 0U; element < num_elements; ++element) {
            if (!ElementContainsNode(element, node)) {
                continue;
            }
            for (auto component = 0U; component < num_dof; ++component) {
                const auto index = row_ptrs(dof_index + component) + current_col[component];
                const auto new_index = ComputeColIndsInElement(element, node, index);
                current_col[component] += new_index - index;
            }
        }

        for (auto constraint = 0U; constraint < num_constraints; ++constraint) {
            if (!ConstraintContainsNode(constraint, dof_index)) {
                continue;
            }
            for (auto component = 0U; component < num_dof; ++component) {
                const auto index = row_ptrs(dof_index + component) + current_col[component];
                const auto new_index = ComputeColIndsInConstraint(constraint, index);
                current_col[component] += new_index - index;
            }
        }
    }
};

}  // namespace openturbine
