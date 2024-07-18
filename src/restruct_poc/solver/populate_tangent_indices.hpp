#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
struct PopulateTangentIndices {
    int num_system_nodes;
    Kokkos::View<int*>::const_type node_state_indices;
    Kokkos::View<int*> indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto entries_so_far = 0;
        for (int i_node = 0; i_node < num_system_nodes; ++i_node) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                const auto i = i_node;
                const auto column_start = node_state_indices(i) * kLieAlgebraComponents;
                for (int m = 0; m < kLieAlgebraComponents; ++m) {
                    indices(entries_so_far) = column_start + m;
                    ++entries_so_far;
                }
            }
        }
    }
};
}  // namespace openturbine
