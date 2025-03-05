#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct ContributeMassesToVector {
    Kokkos::View<size_t* [6]>::const_type element_freedom_table;
    Kokkos::View<double* [6]>::const_type elements;
    Kokkos::View<double* [1], Kokkos::LayoutLeft> vector;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto j = 0U; j < element_freedom_table.extent(1); ++j) {
            vector(element_freedom_table(i_elem, j), 0) += elements(i_elem, j);
        }
    }
};

}  // namespace openturbine
