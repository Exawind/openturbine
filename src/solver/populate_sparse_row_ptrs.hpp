#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename RowPtrType>
struct PopulateSparseRowPtrs {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    RowPtrType row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        const auto num_elems = num_nodes_per_element.extent(0);
        auto rows_so_far = 0;
        for (auto i_elem = 0U; i_elem < num_elems; ++i_elem) {
            auto num_nodes = num_nodes_per_element(i_elem);
            for (auto i = 0U; i < num_nodes * 6U; ++i) {
                row_ptrs(rows_so_far + 1) =
                    row_ptrs(rows_so_far) +
                    static_cast<typename RowPtrType::value_type>(num_nodes * 6U);
                ++rows_so_far;
            }
        }
        auto last_row = rows_so_far;
        for (auto i = last_row + 1; i < row_ptrs.extent_int(0); ++i) {
            row_ptrs(i) = row_ptrs(last_row);
        }
    }
};
}  // namespace openturbine
