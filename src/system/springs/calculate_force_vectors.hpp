#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateForceVectors {
    Kokkos::View<double* [3]>::const_type r_;
    Kokkos::View<double*>::const_type c1_;
    Kokkos::View<double* [3]> f_;

    KOKKOS_FUNCTION
    void operator()(int i_elem) const {
        auto r = Kokkos::subview(r_, i_elem, Kokkos::ALL);
        auto f = Kokkos::subview(f_, i_elem, Kokkos::ALL);

        // Calculate force vector components: f = c1 * r
        for (int i = 0; i < 3; ++i) {
            f(i) = c1_(i_elem) * r(i);
        }
    }
};

}  // namespace openturbine
