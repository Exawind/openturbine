#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
struct PopulateSparseRowPtrs {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    Kokkos::View<int*> row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        const auto num_elems = static_cast<int>(elem_indices.extent(0));
        auto rows_so_far = 0;
        for (int i_elem = 0; i_elem < num_elems; ++i_elem) {
            auto idx = elem_indices[i_elem];
            auto num_nodes = idx.num_nodes;
            for (int i = 0; i < num_nodes * kLieAlgebraComponents; ++i) {
                row_ptrs(rows_so_far + 1) =
                    row_ptrs(rows_so_far) + num_nodes * kLieAlgebraComponents;
                ++rows_so_far;
            }
        }
        auto last_row = rows_so_far;
        for (int i = last_row + 1; i < row_ptrs.extent_int(0); ++i) {
            row_ptrs(i) = row_ptrs(last_row);
        }
    }
};
}  // namespace openturbine