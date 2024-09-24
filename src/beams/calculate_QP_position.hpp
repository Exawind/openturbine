#pragma once

#include <Kokkos_Core.hpp>

#include "src/math/quaternion_operations.hpp"

namespace openturbine {

struct CalculateQPPosition {
    size_t i_elem;
    Kokkos::View<double** [3]>::const_type qp_x0_;
    Kokkos::View<double** [3]>::const_type qp_u_;
    Kokkos::View<double** [4]>::const_type qp_r0_;
    Kokkos::View<double** [4]>::const_type qp_r_;
    Kokkos::View<double** [7]> qp_x_;

    KOKKOS_FUNCTION void operator()(const int i_qp) const {
        // Calculate current position
        qp_x_(i_elem, i_qp, 0) = qp_x0_(i_elem, i_qp, 0) + qp_u_(i_elem, i_qp, 0);
        qp_x_(i_elem, i_qp, 1) = qp_x0_(i_elem, i_qp, 1) + qp_u_(i_elem, i_qp, 1);
        qp_x_(i_elem, i_qp, 2) = qp_x0_(i_elem, i_qp, 2) + qp_u_(i_elem, i_qp, 2);

        // Calculate current orientation
        auto RR0_data = Kokkos::Array<double, 4>{};
        auto RR0 = Kokkos::View<double[4]>(RR0_data.data());
        QuaternionCompose(
            Kokkos::subview(qp_r_, i_elem, i_qp, Kokkos::ALL),
            Kokkos::subview(qp_r0_, i_elem, i_qp, Kokkos::ALL), RR0
        );
        qp_x_(i_elem, i_qp, 3) = RR0(0);
        qp_x_(i_elem, i_qp, 4) = RR0(1);
        qp_x_(i_elem, i_qp, 5) = RR0(2);
        qp_x_(i_elem, i_qp, 6) = RR0(3);
    }
};
}  // namespace openturbine
