#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateConstraintsMatrixFull(
    size_t num_system_dofs,
    size_t num_dofs,
    const Kokkos::View<size_t*>::const_type& base_active_dofs,
    const Kokkos::View<size_t*>::const_type& target_active_dofs,
    const Kokkos::View<size_t* [6]>::const_type& base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& row_range
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full Constraints Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    const auto constraints_row_ptrs = RowPtrType("constraints_row_ptrs", num_dofs + 1);
    const auto constraints_row_entries = RowPtrType("constraints_row_entries", num_dofs);
    Kokkos::parallel_for("ComputeConstraintsRowEntries", row_range.extent(0), KOKKOS_LAMBDA(size_t i_constraint) {
        for (auto i_row = row_range(i_constraint).first; i_row < row_range(i_constraint).second; ++i_row) {
            constraints_row_entries(i_row + num_system_dofs) = base_active_dofs(i_constraint) + target_active_dofs(i_constraint);
        }
    });

    Kokkos::parallel_scan("ComputeConstraintsRowPtrs", num_dofs, KOKKOS_LAMBDA(size_t i, size_t& update, bool is_final) {
        update += constraints_row_entries(i);
        if (is_final) {
            constraints_row_ptrs(i + 1) = update;
        }
    });

    auto constraints_num_non_zero = typename RowPtrType::value_type{};
    Kokkos::deep_copy(constraints_num_non_zero, Kokkos::subview(constraints_row_ptrs, num_dofs));

    const auto constraints_col_inds = IndicesType("constraints_col_inds", constraints_num_non_zero);

    Kokkos::parallel_for("ComputeConstraintsColInds", row_range.extent(0), KOKKOS_LAMBDA(size_t i_constraint) {
        for (auto i_row = row_range(i_constraint).first; i_row < row_range(i_constraint).second; ++i_row) {
            auto current_dof_index = constraints_row_ptrs(i_row + num_system_dofs);
            const auto n_target_node_cols = target_active_dofs(i_constraint);
            for (auto j = 0U; j < n_target_node_cols; ++j, ++current_dof_index) {
                constraints_col_inds(current_dof_index) = static_cast<typename IndicesType::value_type>(target_node_freedom_table(i_constraint, j));
            }

            const auto n_base_node_cols = base_active_dofs(i_constraint);
            for (auto j = 0U; j < n_base_node_cols; ++j, ++current_dof_index) {
                constraints_col_inds(current_dof_index) = static_cast<typename IndicesType::value_type>(base_node_freedom_table(i_constraint, j));
            }
        }
    });

    const auto constraints_values = ValuesType("constraints_values", constraints_num_non_zero);
    KokkosSparse::sort_crs_matrix(constraints_row_ptrs, constraints_col_inds, constraints_values);

    return CrsMatrixType(
        "constraints_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
        constraints_num_non_zero, constraints_values, constraints_row_ptrs,
        constraints_col_inds
    );
}

}  // namespace openturbine
