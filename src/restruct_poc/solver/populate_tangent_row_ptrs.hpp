#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
template <typename size_type>
struct PopulateTangentRowPtrs {
    int num_system_nodes;
    Kokkos::View<size_type*> row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto rows_so_far = 0;
        for (int i_node = 0; i_node < num_system_nodes; ++i_node) {
            for (int i = 0; i < kLieAlgebraComponents; ++i) {
                row_ptrs(rows_so_far + 1) = row_ptrs(rows_so_far) + kLieAlgebraComponents;
                ++rows_so_far;
            }
        }
    }
};
}  // namespace openturbine