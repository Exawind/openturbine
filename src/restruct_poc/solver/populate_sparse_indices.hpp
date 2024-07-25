#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
struct PopulateSparseIndices {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    Kokkos::View<size_t*>::const_type node_state_indices;
    Kokkos::View<int*> indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
        const auto num_elems = elem_indices.extent(0);
        auto entries_so_far = 0;
        for (auto i_elem = 0u; i_elem < num_elems; ++i_elem) {
            auto idx = elem_indices[i_elem];
            auto num_nodes = idx.num_nodes;
            for (auto j_index = 0u; j_index < num_nodes; ++j_index) {
                for (auto n = 0u; n < kLieAlgebraComponents; ++n) {
                    for (auto i_index = 0u; i_index < num_nodes; ++i_index) {
                        const auto i = i_index + idx.node_range.first;
                        const auto column_start = node_state_indices(i) * kLieAlgebraComponents;
                        for (auto m = 0u; m < kLieAlgebraComponents; ++m) {
                            indices(entries_so_far) = static_cast<int>(column_start + m);
                            ++entries_so_far;
                        }
                    }
                }
            }
        }
    }
};
}  // namespace openturbine
