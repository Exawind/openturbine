#pragma once

#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine {

template <typename DeviceType>
struct UpdateGlobalPosition {
    typename Kokkos::View<double* [7], DeviceType>::const_type q;
    typename Kokkos::View<double* [7], DeviceType>::const_type x0;
    Kokkos::View<double* [7], DeviceType> x;

    KOKKOS_FUNCTION
    void operator()(int node) const {
        // Calculate global position
        x(node, 0) = x0(node, 0) + q(node, 0);
        x(node, 1) = x0(node, 1) + q(node, 1);
        x(node, 2) = x0(node, 2) + q(node, 2);

        // Calculate global orientation
        auto node_x_data = Kokkos::Array<double, 4>{};
        const auto node_x = Kokkos::View<double[4], DeviceType>{node_x_data.data()};
        QuaternionCompose(
            Kokkos::subview(q, node, Kokkos::make_pair(3, 7)),
            Kokkos::subview(x0, node, Kokkos::make_pair(3, 7)), node_x
        );
        x(node, 3) = node_x(0);
        x(node, 4) = node_x(1);
        x(node, 5) = node_x(2);
        x(node, 6) = node_x(3);
    }
};

}  // namespace openturbine
