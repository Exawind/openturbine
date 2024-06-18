#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct PopulateSparseIndices_Constraints_Transpose {
    int num_constraint_nodes;
    Kokkos::View<Constraints::NodeIndices*> node_indices;
    Kokkos::View<int*> B_indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto entries_so_far = 0;
        for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
            auto i_col = i_constraint * kLieAlgebraComponents;
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                for (int j = 0; j < kLieAlgebraComponents; ++j) {
                    B_indices(entries_so_far) = i_col + j;
                    ++entries_so_far;
                }
            }
        }
    }
};

}  // namespace openturbine
