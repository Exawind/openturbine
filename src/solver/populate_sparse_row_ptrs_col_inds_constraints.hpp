#pragma once

#include <Kokkos_Core.hpp>

#include "src/constraints/constraint_type.hpp"

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct PopulateSparseRowPtrsColInds_Constraints {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t* [6]>::const_type base_node_freedom_table;
    Kokkos::View<size_t* [6]>::const_type target_node_freedom_table;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type constraint_base_node_col_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type constraint_target_node_col_range;
    RowPtrType B_row_ptrs;
    IndicesType B_col_inds;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto ind_col = 0U;
        for (auto i_constraint = 0U; i_constraint < type.extent(0); ++i_constraint) {
            // Loop through rows that apply to this constraint
            for (auto i_row = row_range(i_constraint).first; i_row < row_range(i_constraint).second;
                 ++i_row) {
                // Set first column index in this row
                B_row_ptrs(i_row) = ind_col;

                // Add column indices for target node
                const auto n_target_node_cols =
                    constraint_target_node_col_range(i_constraint).second -
                    constraint_target_node_col_range(i_constraint).first;
                for (auto j = 0U; j < n_target_node_cols; ++j) {
                    B_col_inds(ind_col) = static_cast<typename IndicesType::value_type>(
                        target_node_freedom_table(i_constraint, j)
                    );
                    ind_col++;
                }

                // Add column indices for base node
                const auto n_base_node_cols = constraint_base_node_col_range(i_constraint).second -
                                              constraint_base_node_col_range(i_constraint).first;
                for (auto j = 0U; j < n_base_node_cols; ++j) {
                    B_col_inds(ind_col) = static_cast<typename IndicesType::value_type>(
                        base_node_freedom_table(i_constraint, j)
                    );
                    ind_col++;
                }
            }
        }
        // Set final column index
        B_row_ptrs(B_row_ptrs.extent_int(0) - 1) = ind_col;
    }
};

}  // namespace openturbine
