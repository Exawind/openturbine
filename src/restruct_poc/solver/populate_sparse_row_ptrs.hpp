#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
template <typename RowPtrType>
struct PopulateSparseRowPtrs {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    RowPtrType row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        const auto num_elems = static_cast<int>(elem_indices.extent(0));
        auto rows_so_far = 0;
        for (int i_elem = 0; i_elem < num_elems; ++i_elem) {
            auto idx = elem_indices[i_elem];
            auto num_nodes = idx.num_nodes;
            for (auto i = 0u; i < num_nodes * kLieAlgebraComponents; ++i) {
                row_ptrs(rows_so_far + 1) =
                    row_ptrs(rows_so_far) +
                    static_cast<typename RowPtrType::value_type>(num_nodes * kLieAlgebraComponents);
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