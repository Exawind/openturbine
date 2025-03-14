#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct ComputeConstraintsColInds {
    using RowPtrValueType = typename RowPtrType::value_type;
    using IndicesValueType = typename IndicesType::value_type;
    size_t num_system_dofs{};
    Kokkos::View<size_t*>::const_type base_active_dofs;
    Kokkos::View<size_t*>::const_type target_active_dofs;
    Kokkos::View<size_t* [6]>::const_type base_node_freedom_table;
    Kokkos::View<size_t* [6]>::const_type target_node_freedom_table;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    typename RowPtrType::const_type row_ptrs;
    IndicesType col_inds;

    KOKKOS_FUNCTION
    RowPtrValueType CalculateTargetInds(size_t constraint, RowPtrValueType dof_index) const {
        const auto target_cols = target_active_dofs(constraint);
        for (auto j = 0U; j < target_cols; ++j, ++dof_index) {
            const auto index = target_node_freedom_table(constraint, j);
            col_inds(dof_index) = static_cast<IndicesValueType>(index);
        }
        return dof_index;
    }

    KOKKOS_FUNCTION
    RowPtrValueType CalculateBaseInds(size_t constraint, RowPtrValueType dof_index) const {
        const auto base_cols = base_active_dofs(constraint);
        for (auto j = 0U; j < base_cols; ++j, ++dof_index) {
            const auto index = base_node_freedom_table(constraint, j);
            col_inds(dof_index) = static_cast<IndicesValueType>(index);
        }
        return dof_index;
    }

    KOKKOS_FUNCTION
    void operator()(size_t constraint) const {
        const auto first_row = row_range(constraint).first;
        const auto end_row = row_range(constraint).second;
        for (auto row = first_row; row < end_row; ++row) {
            const auto initial_index = row_ptrs(row + num_system_dofs);
            const auto second_index = CalculateTargetInds(constraint, initial_index);
            CalculateBaseInds(constraint, second_index);
        }
    }
};

}  // namespace openturbine
