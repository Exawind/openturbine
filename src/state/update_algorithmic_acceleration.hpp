#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct UpdateAlgorithmicAcceleration {
    Kokkos::View<double* [6]> acceleration;
    Kokkos::View<double* [6]>::const_type vd;
    double alpha_f;
    double alpha_m;

    KOKKOS_FUNCTION
    void operator()(int i) const {
        for (int j = 0; j < 6; ++j) {
            acceleration(i, j) += (1. - alpha_f) / (1. - alpha_m) * vd(i, j);
        }
    }
};

}  // namespace openturbine
