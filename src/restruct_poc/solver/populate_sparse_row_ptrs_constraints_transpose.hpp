#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {
struct PopulateSparseRowPtrs_Constraints_Transpose {
    int num_constraint_nodes;
    int num_system_nodes;
    Kokkos::View<Constraints::DeviceData*> data;
    Kokkos::View<int*> B_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto cum_cols = 0;
        for (int i_node = 0; i_node < num_system_nodes; ++i_node) {
            int num_blocks = 0;
            for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
                num_blocks += (data(i_constraint).target_node_index == i_node) ? 1 : 0;
                num_blocks += (data(i_constraint).base_node_index == i_node) ? 1 : 0;
            }
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                B_row_ptrs(i_node * kLieAlgebraComponents + i) = cum_cols;
                cum_cols += num_blocks * kLieAlgebraComponents;
            }
        }
        B_row_ptrs(num_system_nodes * kLieAlgebraComponents) = cum_cols;
    }
};
}  // namespace openturbine