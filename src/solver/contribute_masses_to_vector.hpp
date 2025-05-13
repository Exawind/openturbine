#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct ContributeMassesToVector {
    typename Kokkos::View<size_t* [6], DeviceType>::const_type element_freedom_table;
    typename Kokkos::View<double* [6], DeviceType>::const_type elements;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> vector;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto j = 0U; j < element_freedom_table.extent(1); ++j) {
            vector(element_freedom_table(i_elem, j), 0) += elements(i_elem, j);
        }
    }
};

}  // namespace openturbine
