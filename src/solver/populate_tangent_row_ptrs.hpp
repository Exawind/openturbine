#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename size_type>
struct PopulateTangentRowPtrs {
    size_t num_system_nodes;
    Kokkos::View<size_type*> row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
        auto rows_so_far = 0;
        for (auto i_node = 0U; i_node < num_system_nodes; ++i_node) {
            for (auto i = 0U; i < 6U; ++i) {
                row_ptrs(rows_so_far + 1) = row_ptrs(rows_so_far) + 6U;
                ++rows_so_far;
            }
        }
    }
};
}  // namespace openturbine
