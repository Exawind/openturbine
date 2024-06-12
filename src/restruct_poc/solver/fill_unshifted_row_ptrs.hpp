#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct FillUnshiftedRowPtrs {
    Kokkos::View<int*> new_row_ptrs;
    int num_system_dofs;
    Kokkos::View<const int*> old_row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int i) const {
        auto last_row_map_index = num_system_dofs+1;
        if(i < last_row_map_index) {
            new_row_ptrs(i) = old_row_ptrs(i);
        }
        else {
            new_row_ptrs(i) = old_row_ptrs(num_system_dofs);
        }
    }

};

}