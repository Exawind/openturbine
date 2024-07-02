#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {
struct PopulateSparseRowPtrs_Constraints {
    int num_constraint_nodes;
    Kokkos::View<Constraints::Data*> data;
    Kokkos::View<int*> B_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto cum_cols = 0;
        for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
            // Get reference to constraint data for this constraint
            auto& cd = data(i_constraint);

            // Number of blocks in row
            auto num_blocks_in_row = cd.base_node_index < 0 ? 1 : 2;

            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                // Row index
                auto row_index = i_constraint * kLieAlgebraComponents + i;

                // Cumulative number of columns
                B_row_ptrs(row_index) = cum_cols;
                cum_cols += num_blocks_in_row * kLieAlgebraComponents;
            }
        }
        B_row_ptrs(num_constraint_nodes * kLieAlgebraComponents) = cum_cols;
    }
};
}  // namespace openturbine