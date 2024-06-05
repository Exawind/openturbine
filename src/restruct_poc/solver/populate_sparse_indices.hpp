#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
struct PopulateSparseIndices {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    Kokkos::View<int*>::const_type node_state_indices;
    Kokkos::View<int*> indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
        const auto num_elems = static_cast<int>(elem_indices.extent(0));
        auto entries_so_far = 0;
        for (int i_elem = 0; i_elem < num_elems; ++i_elem) {
            auto idx = elem_indices[i_elem];
            auto num_nodes = idx.num_nodes;
            for (int j_index = 0; j_index < num_nodes; ++j_index) {
                for (int n = 0; n < kLieAlgebraComponents; ++n) {
                    for (int i_index = 0; i_index < num_nodes; ++i_index) {
                        const auto i = i_index + idx.node_range.first;
                        const auto column_start = node_state_indices(i) * kLieAlgebraComponents;
                        for (int m = 0; m < kLieAlgebraComponents; ++m) {
                            indices(entries_so_far) = column_start + m;
                            ++entries_so_far;
                        }
                    }
                }
            }
        }
    }
};
}  // namespace openturbine
