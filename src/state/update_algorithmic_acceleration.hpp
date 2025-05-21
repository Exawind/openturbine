#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct UpdateAlgorithmicAcceleration {
    typename Kokkos::View<double* [6], DeviceType> acceleration;
    typename Kokkos::View<double* [6], DeviceType>::const_type vd;
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
