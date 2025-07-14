#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct CalculateNextState {
    double h;
    double alpha_f;
    double alpha_m;
    double beta;
    double gamma;
    Kokkos::View<double* [6], DeviceType> q_delta;
    Kokkos::View<double* [6], DeviceType> v;
    Kokkos::View<double* [6], DeviceType> vd;
    Kokkos::View<double* [6], DeviceType> a;

    KOKKOS_FUNCTION
    void operator()(const size_t node) const {
        for (auto component = 0U; component < 6U; ++component) {
            const double v_p = v(node, component);    // Save velocity from previous iteration
            const double vd_p = vd(node, component);  // Save acceleration from previous iteration
            const double a_p =
                a(node, component);  // Save algorithmic acceleration from previous iteration
            vd(node, component) = 0.;
            a(node, component) = (alpha_f * vd_p - alpha_m * a_p) / (1. - alpha_m);
            v(node, component) = v_p + h * (1. - gamma) * a_p + gamma * h * a(node, component);
            q_delta(node, component) = v_p + (0.5 - beta) * h * a_p + beta * h * a(node, component);
        }
    }
};

}  // namespace openturbine
