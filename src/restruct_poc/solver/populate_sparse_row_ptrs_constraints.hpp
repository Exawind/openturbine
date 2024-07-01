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
            int factor = 0;
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                switch (data(i_constraint).type) {
                    case ConstraintType::FixedBC:
                    case ConstraintType::PrescribedBC:
                        factor = 1;
                        break;
                    default:
                        factor = 2;
                }
                B_row_ptrs(rows_so_far + 1) =
                    B_row_ptrs(rows_so_far) + factor * kLieAlgebraComponents;
                ++rows_so_far;
            }
        }
    }
};
}  // namespace openturbine