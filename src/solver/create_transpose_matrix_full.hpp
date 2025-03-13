#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateTransposeMatrixFull(
    size_t num_system_dofs,
    size_t num_dofs,
    const Kokkos::View<size_t*>::const_type& active_dofs,
    const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
    const Kokkos::View<size_t*>::const_type& base_active_dofs,
    const Kokkos::View<size_t*>::const_type& target_active_dofs,
    const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full Transpose Constraints Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    const auto constraints_row_ptrs = RowPtrType("row_ptrs", num_dofs + 1);
    const auto constraints_row_entries = RowPtrType("row_entries", num_dofs);

    Kokkos::parallel_for("ComputeTransposeConstraintsRowEntries", active_dofs.extent(0), KOKKOS_LAMBDA(size_t i) {
        const auto this_node_num_dof = active_dofs(i);
        const auto this_node_dof_index = node_freedom_map_table(i);

        auto num_entries_in_row = 0UL;
        for (auto i_constraint = 0U; i_constraint < constraint_row_range.extent(0); ++i_constraint) {
            const auto num_columns = constraint_row_range(i_constraint).second - constraint_row_range(i_constraint).first;
            if(this_node_dof_index == constraint_base_node_freedom_table(i_constraint, 0) && (base_active_dofs(i_constraint) != 0UL)) {
                    num_entries_in_row += num_columns;
            }
            else if(this_node_dof_index == constraint_target_node_freedom_table(i_constraint, 0) && (target_active_dofs(i_constraint) != 0UL)) {
                    num_entries_in_row += num_columns;
            }
        }

        for (auto j = 0U; j < this_node_num_dof; ++j) {
            constraints_row_entries(this_node_dof_index + j) = num_entries_in_row;
        }
    });

    Kokkos::parallel_scan("ComputeTransposeConstraintsRowPtrs", num_dofs, KOKKOS_LAMBDA(size_t i, size_t& update, bool is_final) {
        update += constraints_row_entries(i);
        if (is_final) {
            constraints_row_ptrs(i + 1) = update;
        }
    });

    auto constraints_num_non_zero = typename RowPtrType::value_type{};
    Kokkos::deep_copy(constraints_num_non_zero, Kokkos::subview(constraints_row_ptrs, num_dofs));

    auto constraints_col_inds = IndicesType("col_inds", constraints_num_non_zero);

    Kokkos::parallel_for("ComputeTransposeConstraintsColInds", active_dofs.extent(0), KOKKOS_LAMBDA(size_t i) {
        const auto this_node_num_dof = active_dofs(i);
        const auto this_node_dof_index = node_freedom_map_table(i);
        auto current_col = Kokkos::Array<typename RowPtrType::value_type, 6>{};

        for (auto i_constraint = 0U; i_constraint < constraint_row_range.extent(0); ++i_constraint) {
            if(this_node_dof_index == constraint_base_node_freedom_table(i_constraint, 0) && (base_active_dofs(i_constraint) != 0UL)) {
                for (auto j = 0U; j < this_node_num_dof; ++j) {
                    const auto current_dof_index = constraints_row_ptrs(this_node_dof_index + j) + current_col[j];
                    auto new_dof_index = current_dof_index;;
                    for (auto k = constraint_row_range(i_constraint).first; k < constraint_row_range(i_constraint).second; ++k, ++new_dof_index) {
                        constraints_col_inds(new_dof_index) = static_cast<typename IndicesType::value_type>(num_system_dofs + k);
                    }
                    current_col[j] += new_dof_index - current_dof_index;
                }
            }
            if(this_node_dof_index == constraint_target_node_freedom_table(i_constraint, 0) && (target_active_dofs(i_constraint) != 0UL)) {
                for (auto j = 0U; j < this_node_num_dof; ++j) {
                    const auto current_dof_index = constraints_row_ptrs(this_node_dof_index + j) + current_col[j];
                    auto new_dof_index = current_dof_index;;
                    for (auto k = constraint_row_range(i_constraint).first; k < constraint_row_range(i_constraint).second; ++k, ++new_dof_index) {
                        constraints_col_inds(new_dof_index) = static_cast<typename IndicesType::value_type>(num_system_dofs + k);
                    }
                    current_col[j] += new_dof_index - current_dof_index;
                }
            }
        
        }
    });

    const auto constraints_values = ValuesType("constraints_values", constraints_num_non_zero);

    KokkosSparse::sort_crs_matrix(constraints_row_ptrs, constraints_col_inds, constraints_values);

    return CrsMatrixType(
        "transpose_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs), constraints_num_non_zero,
        constraints_values, constraints_row_ptrs, constraints_col_inds
    );
}

}  // namespace openturbine
