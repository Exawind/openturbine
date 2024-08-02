#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct PopulateSparseRowPtrsColInds_Constraints {
    Kokkos::View<Constraints::DeviceData*> data;
    RowPtrType B_row_ptrs;
    IndicesType B_col_inds;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto ind_col = 0U;
        for (auto i_constraint = 0U; i_constraint < data.extent(0); ++i_constraint) {
            // Get reference to constraint data for this constraint
            auto& cd = data(i_constraint);

            // Loop through rows that apply to this constraint
            for (auto i_row = cd.row_range.first; i_row < cd.row_range.second; ++i_row) {
                // Set first column index in this row
                B_row_ptrs(i_row) = ind_col;

                // Add column indices for target node
                for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
                    B_col_inds(ind_col) = static_cast<typename IndicesType::value_type>(
                        static_cast<size_t>(cd.target_node_index) * kLieAlgebraComponents + j
                    );
                    ind_col++;
                }

                // Add column indices for base node if it has a valid index
                if (cd.base_node_index >= 0) {
                    // Add column indices for base node
                    for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
                        B_col_inds(ind_col) = static_cast<typename IndicesType::value_type>(
                            static_cast<size_t>(cd.base_node_index) * kLieAlgebraComponents + j
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
