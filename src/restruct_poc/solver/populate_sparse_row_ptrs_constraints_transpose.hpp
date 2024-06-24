#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {
struct PopulateSparseRowPtrs_Constraints_Transpose {
    int num_constraint_nodes;
    int num_system_nodes;
    Kokkos::View<Constraints::NodeIndices*> node_indices;
    Kokkos::View<unsigned*> B_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto rows_so_far = 0;
        for (int i_system = 0; i_system < num_system_nodes; ++i_system) {
            bool is_constraint_node = false;
            for (int i_constraint = 0; i_constraint < num_constraint_nodes; ++i_constraint) {
                is_constraint_node = is_constraint_node ||
                                     (node_indices(i_constraint).constrained_node_index == i_system);
            }
            auto row_entries = (is_constraint_node) ? kLieAlgebraComponents : 0;
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                B_row_ptrs(rows_so_far + 1) = B_row_ptrs(rows_so_far) + row_entries;
                ++rows_so_far;
            }
        }
    }
};
}  // namespace openturbine