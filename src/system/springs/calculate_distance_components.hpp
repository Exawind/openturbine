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

        r(0) = x0(0) + u2(0) - u1(0);
        r(1) = x0(1) + u2(1) - u1(1);
        r(2) = x0(2) + u2(2) - u1(2);
    }
};

}  // namespace openturbine
