#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {
struct PopulateSparseRowPtrs_Constraints_Transpose {
    int num_constraint_nodes;
    int num_system_nodes;
    Kokkos::View<Constraints::Data*> data;
    Kokkos::View<int*> B_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto rows_so_far = 0;
        for (int i_system = 0; i_system < num_system_nodes; ++i_system) {
            int num_blocks = 0;
            for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
                num_blocks += (data(i_constraint).target_node_index == i_system) ? 1 : 0;
                num_blocks += (data(i_constraint).base_node_index == i_system) ? 1 : 0;
            }
            auto row_entries = num_blocks * kLieAlgebraComponents;
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                B_row_ptrs(rows_so_far + 1) = B_row_ptrs(rows_so_far) + row_entries;
                ++rows_so_far;
            }
        }
    }
};
}  // namespace openturbine