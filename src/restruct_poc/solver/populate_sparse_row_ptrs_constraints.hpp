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
        auto rows_so_far = 0;
        for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                auto num_cols =
                    (data(i_constraint).base_node_index < 0 ? 1 : 2) * kLieAlgebraComponents;
                B_row_ptrs(rows_so_far + 1) = B_row_ptrs(rows_so_far) + num_cols;
                ++rows_so_far;
            }
        }
    }
};
}  // namespace openturbine