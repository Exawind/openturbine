#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/constraints/constraint_type.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct PopulateSparseRowPtrsColInds_Constraints {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
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
                for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
                    B_col_inds(ind_col) = static_cast<typename IndicesType::value_type>(
                        target_node_index(i_constraint) * kLieAlgebraComponents + j
                    );
                    ind_col++;
                }

                // Add column indices for base node if it has a valid index
                if (GetNumberOfNodes(type(i_constraint)) == 2U) {
                    // Add column indices for base node
                    for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
                        B_col_inds(ind_col) = static_cast<typename IndicesType::value_type>(
                            base_node_index(i_constraint) * kLieAlgebraComponents + j
                        );
                        ind_col++;
                    }
                }
            }
        }
        // Set final column index
        B_row_ptrs(B_row_ptrs.extent_int(0) - 1) = ind_col;
    }
};

}  // namespace openturbine
