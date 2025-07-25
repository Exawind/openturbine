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
    void operator()(int node) const {
        for (auto component = 0; component < 6; ++component) {
            acceleration(node, component) += (1. - alpha_f) / (1. - alpha_m) * vd(node, component);
        }
    }
};

}  // namespace openturbine
