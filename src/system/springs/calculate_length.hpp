#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateLength {
    Kokkos::View<double* [3]>::const_type r_;
    Kokkos::View<double*> l_;

    KOKKOS_FUNCTION
    void operator()(int i_elem) const {
        auto r = Kokkos::subview(r_, i_elem, Kokkos::ALL);
        l_(i_elem) = sqrt(r(0) * r(0) + r(1) * r(1) + r(2) * r(2));
    }
};

}  // namespace openturbine
