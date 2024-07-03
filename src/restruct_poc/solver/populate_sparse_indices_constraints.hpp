#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct PopulateSparseIndices_Constraints {
    int num_constraint_nodes;
    Kokkos::View<Constraints::Data*> data;
    Kokkos::View<int*> B_col_inds;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto ind_num = 0;
        for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
            // Get reference to constraint data for this constraint
            auto& cd = data(i_constraint);

            for (int j = 0; j < kLieAlgebraComponents; ++j) {
                for (int i = 0; i < kLieAlgebraComponents; ++i) {
                    B_col_inds(ind_num) = cd.target_node_index * kLieAlgebraComponents + i;
                    ind_num++;
                }

                if (cd.base_node_index >= 0) {
                    for (int i = 0; i < kLieAlgebraComponents; ++i) {
                        B_col_inds(ind_num) = cd.base_node_index * kLieAlgebraComponents + i;
                        ind_num++;
                    }
                }
            }
        }
    }
};

}  // namespace openturbine
