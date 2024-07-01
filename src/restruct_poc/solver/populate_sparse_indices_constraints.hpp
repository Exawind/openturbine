#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct PopulateSparseIndices_Constraints {
    int num_constraint_nodes;
    Kokkos::View<Constraints::Data*> data;
    Kokkos::View<int*> B_indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto entries_so_far = 0;
        for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
            // Get reference to constraint data for this constraint
            auto& cd = data(i_constraint);

            // Target node block
            auto i_col = cd.target_node_index * kLieAlgebraComponents;
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                for (int j = 0; j < kLieAlgebraComponents; ++j) {
                    B_indices(entries_so_far) = i_col + j;
                    ++entries_so_far;
                }
            }

            // Base node block if column is positive
            i_col = cd.base_node_index * kLieAlgebraComponents;
            if (i_col >= 0) {
                for (int i = 0; i < kLieAlgebraComponents; ++i) {
                    for (int j = 0; j < kLieAlgebraComponents; ++j) {
                        B_indices(entries_so_far) = i_col + j;
                        ++entries_so_far;
                    }
                }
            }
        }
    }
};

}  // namespace openturbine
