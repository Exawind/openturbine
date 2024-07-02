#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct PopulateSparseIndices_Constraints_Transpose {
    int num_constraint_nodes;
    int num_system_nodes;
    Kokkos::View<Constraints::Data*> data;
    Kokkos::View<int*> B_col_inds;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto col_entries = 0;
        for (int i_node = 0; i_node < num_system_nodes; ++i_node) {
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
                    if (data(i_constraint).target_node_index == i_node ||
                        data(i_constraint).base_node_index == i_node) {
                        for (int j = 0; j < kLieAlgebraComponents; ++j) {
                            B_col_inds(col_entries) = i_constraint * kLieAlgebraComponents + j;
                            ++col_entries;
                        }
                    }
                }
            }
        }
    }
};

}  // namespace openturbine
