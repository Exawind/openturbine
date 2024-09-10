#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {
struct PopulateSparseIndices {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<int*> indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
        const auto num_elems = num_nodes_per_element.extent(0);
        auto entries_so_far = 0;
        for (auto i_elem = 0U; i_elem < num_elems; ++i_elem) {
            auto num_nodes = num_nodes_per_element(i_elem);
            for (auto j_index = 0U; j_index < num_nodes; ++j_index) {
                for (auto n = 0U; n < 6U; ++n) {
                    for (auto i_index = 0U; i_index < num_nodes; ++i_index) {
                        const auto column_start = node_state_indices(i_elem, i_index) * 6U;
                        for (auto m = 0U; m < 6U; ++m) {
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
