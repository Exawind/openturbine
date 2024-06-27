#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {
template <typename size_type>
struct PopulateSparseRowPtrs_Constraints {
    int num_constraint_nodes;
    Kokkos::View<size_type*> B_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto rows_so_far = 0;
        for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                B_row_ptrs(rows_so_far + 1) = B_row_ptrs(rows_so_far) + kLieAlgebraComponents;
                ++rows_so_far;
            }
        }
    }
};
}  // namespace openturbine