#pragma once

#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine {

struct CalculateQPDeformation {
    Kokkos::View<size_t*>::const_type qp_num_qps_per_element;
    Kokkos::View<double** [3]>::const_type qp_x0_;
    Kokkos::View<double** [4]>::const_type qp_r_;
    Kokkos::View<double** [7]>::const_type qp_x_;
    Kokkos::View<double** [3]> qp_deformation_;

    KOKKOS_FUNCTION void operator()(const size_t element, const size_t qp) const {
        // Exit if quadrature point is too high
        if (qp >= qp_num_qps_per_element(element)) {
            return;
        }

        // Element root node initial position
        auto x0_root_data = Kokkos::Array<double, 3>{
            qp_x0_(element, 0, 0),
            qp_x0_(element, 0, 1),
            qp_x0_(element, 0, 2),
        };
        auto x0_root = Kokkos::View<double[3]>::const_type(x0_root_data.data());

        // Element root node current position
        auto x_root_data = Kokkos::Array<double, 3>{
            qp_x_(element, 0, 0),
            qp_x_(element, 0, 1),
            qp_x_(element, 0, 2),
        };
        auto x_root = Kokkos::View<double[3]>::const_type(x_root_data.data());

        // Element root node rotational displacement
        auto r_root_data = Kokkos::Array<double, 4>{
            qp_r_(element, 0, 0),
            qp_r_(element, 0, 1),
            qp_r_(element, 0, 2),
            qp_r_(element, 0, 3),
        };
        auto r_root = Kokkos::View<double[4]>::const_type(r_root_data.data());

        // QP current location
        auto x_data = Kokkos::Array<double, 3>{
            qp_x_(element, qp, 0),
            qp_x_(element, qp, 1),
            qp_x_(element, qp, 2),
        };
        auto x = Kokkos::View<double[3]>::const_type(x_data.data());

        // Initial distance from root to qp
        auto dx0_data = Kokkos::Array<double, 3>{
            qp_x0_(element, qp, 0) - x0_root(0),
            qp_x0_(element, qp, 1) - x0_root(1),
            qp_x0_(element, qp, 2) - x0_root(2),
        };
        auto dx0 = Kokkos::View<double[3]>::const_type(dx0_data.data());

        // Distance from root to qp after applying root rotation
        auto dx_data = Kokkos::Array<double, 3>{};
        auto dx = Kokkos::View<double[3]>(dx_data.data());
        RotateVectorByQuaternion(r_root, dx0, dx);

        // Undeformed quadrature point position
        auto x_undeformed_data = Kokkos::Array<double, 3>{
            dx(0) + x_root(0),
            dx(1) + x_root(1),
            dx(2) + x_root(2),
        };
        auto x_undeformed = Kokkos::View<double[3]>::const_type(x_undeformed_data.data());

        // Calculate difference between current position and undeformed position
        qp_deformation_(element, qp, 0) = x(0) - x_undeformed(0);
        qp_deformation_(element, qp, 1) = x(1) - x_undeformed(1);
        qp_deformation_(element, qp, 2) = x(2) - x_undeformed(2);
    }
};

}  // namespace openturbine
