#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateDistanceComponents {
    Kokkos::View<double* [3]>::const_type x0_;
    Kokkos::View<double* [3]>::const_type u1_;
    Kokkos::View<double* [3]>::const_type u2_;
    Kokkos::View<double* [3]> r_;

    KOKKOS_FUNCTION
    void operator()(int i_elem) const {
        auto x0 = Kokkos::subview(x0_, i_elem, Kokkos::ALL);
        auto u1 = Kokkos::subview(u1_, i_elem, Kokkos::ALL);
        auto u2 = Kokkos::subview(u2_, i_elem, Kokkos::ALL);
        auto r = Kokkos::subview(r_, i_elem, Kokkos::ALL);

        for (int i = 0; i < 3; ++i) {
            r(i) = x0(i) + u2(i) - u1(i);
        }
    }
};

}  // namespace openturbine
