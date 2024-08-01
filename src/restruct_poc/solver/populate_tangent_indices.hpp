#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
struct PopulateTangentIndices {
    size_t num_system_nodes;
    Kokkos::View<int*>::const_type node_state_indices;
    Kokkos::View<int*> indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto entries_so_far = 0U;
        for (auto i_node = 0U; i_node < num_system_nodes; ++i_node) {
            for (auto n = 0U; n < kLieAlgebraComponents; ++n) {
                const auto i = i_node;
                const auto column_start =
                    static_cast<size_t>(node_state_indices(i)) * kLieAlgebraComponents;
                for (auto m = 0U; m < kLieAlgebraComponents; ++m) {
                    indices(entries_so_far) = static_cast<int>(column_start + m);
                    ++entries_so_far;
                }
            }
        }
    }
};
}  // namespace openturbine
