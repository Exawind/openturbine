#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct ContributeSpringsToVector {
    Kokkos::View<size_t* [2][3]>::const_type element_freedom_table;
    Kokkos::View<double* [2][3]>::const_type elements;
    Kokkos::View<double*> vector;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        constexpr auto force_atomic = !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>;
        for (auto j = 0U; j < element_freedom_table.extent(2); ++j) {
            if constexpr (force_atomic) {
                Kokkos::atomic_add(
                    &vector(element_freedom_table(i_elem, 0, j)), elements(i_elem, 0, j)
                );
                Kokkos::atomic_add(
                    &vector(element_freedom_table(i_elem, 1, j)), elements(i_elem, 1, j)
                );
            } else {
                vector(element_freedom_table(i_elem, 0, j)) += elements(i_elem, 0, j);
                vector(element_freedom_table(i_elem, 1, j)) += elements(i_elem, 1, j);
            }
        }
    };
};

}  // namespace openturbine
