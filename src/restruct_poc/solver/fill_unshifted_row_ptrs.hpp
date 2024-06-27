#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename new_size_type, typename old_size_type>
struct FillUnshiftedRowPtrs {
    Kokkos::View<new_size_type*> new_row_ptrs;
    int num_system_dofs;
    Kokkos::View<const old_size_type*> old_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int i) const {
        auto last_row_map_index = num_system_dofs + 1;
        if (i < last_row_map_index) {
            new_row_ptrs(i) = old_row_ptrs(i);
        } else {
            new_row_ptrs(i) = old_row_ptrs(num_system_dofs);
        }
    }
};

}  // namespace openturbine