#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateForceCalculationCoefficients {
    Kokkos::View<double*> c1_;
    Kokkos::View<double*> c2_;
    Kokkos::View<double*>::const_type k_;
    Kokkos::View<double*>::const_type l_ref_;
    Kokkos::View<double*>::const_type l_;

    KOKKOS_FUNCTION
    void operator()(int i_elem) const {
        // c1 = k * (l_ref/l - 1)
        c1_(i_elem) = k_(i_elem) * (l_ref_(i_elem) / l_(i_elem) - 1.0);
        // c2 = k * l_ref/(l^3)
        c2_(i_elem) = k_(i_elem) * l_ref_(i_elem) / (l_(i_elem) * l_(i_elem) * l_(i_elem));
    }
};

}  // namespace openturbine
