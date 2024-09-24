#pragma once

#include <Kokkos_Core.hpp>

#include "src/math/quaternion_operations.hpp"

namespace openturbine {

struct UpdateGlobalPosition {
    Kokkos::View<double* [7]>::const_type q;
    Kokkos::View<double* [7]>::const_type x0;
    Kokkos::View<double* [7]> x;

    KOKKOS_FUNCTION
    void operator()(int i) const {
        // Calculate global position
        x(i, 0) = x0(i, 0) + q(i, 0);
        x(i, 1) = x0(i, 1) + q(i, 1);
        x(i, 2) = x0(i, 2) + q(i, 2);

        // Calculate global orientation
        auto node_x_data = Kokkos::Array<double, 4>{};
        const auto node_x = Kokkos::View<double[4]>{node_x_data.data()};
        QuaternionCompose(
            Kokkos::subview(q, i, Kokkos::make_pair(3, 7)),
            Kokkos::subview(x0, i, Kokkos::make_pair(3, 7)), node_x
        );
        x(i, 3) = node_x(0);
        x(i, 4) = node_x(1);
        x(i, 5) = node_x(2);
        x(i, 6) = node_x(3);
    }
};

}  // namespace openturbine
