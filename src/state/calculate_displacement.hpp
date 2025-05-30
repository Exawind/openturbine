#pragma once

#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine {

template <typename DeviceType>
struct CalculateDisplacement {
    double h;
    typename Kokkos::View<double* [6], DeviceType>::const_type q_delta;
    typename Kokkos::View<double* [7], DeviceType>::const_type q_prev;
    Kokkos::View<double* [7], DeviceType> q;

    KOKKOS_FUNCTION
    void operator()(const int i_node) const {
        for (int i = 0; i < 3; ++i) {
            q(i_node, i) = q_prev(i_node, i) + h * q_delta(i_node, i);
        }

        auto delta_data = Kokkos::Array<double, 3>{
            h * q_delta(i_node, 3), h * q_delta(i_node, 4), h * q_delta(i_node, 5)
        };
        auto delta = Kokkos::View<double[3], DeviceType>{delta_data.data()};

        auto quat_delta_data = Kokkos::Array<double, 4>{};
        auto quat_delta = Kokkos::View<double[4], DeviceType>{quat_delta_data.data()};
        RotationVectorToQuaternion(delta, quat_delta);
        auto quat_prev_data = Kokkos::Array<double, 4>{
            q_prev(i_node, 3), q_prev(i_node, 4), q_prev(i_node, 5), q_prev(i_node, 6)
        };
        auto quat_prev = Kokkos::View<double[4], DeviceType>{quat_prev_data.data()};
        auto quat_new_data = Kokkos::Array<double, 4>{};
        auto quat_new = Kokkos::View<double[4], DeviceType>{quat_new_data.data()};
        QuaternionCompose(quat_delta, quat_prev, quat_new);

        for (int i = 0; i < 4; ++i) {
            q(i_node, i + 3) = quat_new(i);
        }
    }
};

}  // namespace openturbine
