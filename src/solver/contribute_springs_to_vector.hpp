#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct ContributeSpringsToVector {
    Kokkos::View<size_t* [2][3]>::const_type element_freedom_table;
    Kokkos::View<double* [2][3]>::const_type elements;
    Kokkos::View<double*> vector;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto i_node = 0U; i_node < 2U; ++i_node) {
            for (auto j = 0U; j < element_freedom_table.extent(2); ++j) {
                vector(element_freedom_table(i_elem, i_node, j)) += elements(i_elem, i_node, j);
            }
        }
    }
};

}  // namespace openturbine
