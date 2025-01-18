#pragma once

#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine {

struct CalculateRR0 {
    size_t i_elem;
    Kokkos::View<double** [7]>::const_type qp_x_;
    Kokkos::View<double** [6][6]> qp_RR0_;

    KOKKOS_FUNCTION void operator()(const int i_qp) const {
        auto RR0_quaternion_data = Kokkos::Array<double, 4>{
            qp_x_(i_elem, i_qp, 3), qp_x_(i_elem, i_qp, 4), qp_x_(i_elem, i_qp, 5),
            qp_x_(i_elem, i_qp, 6)
        };
        auto RR0_quaternion = Kokkos::View<double[4]>::const_type(RR0_quaternion_data.data());
        auto RR0_data = Kokkos::Array<double, 9>{};
        auto RR0 = View_3x3(RR0_data.data());
        QuaternionToRotationMatrix(RR0_quaternion, RR0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                qp_RR0_(i_elem, i_qp, i, j) = RR0(i, j);
                qp_RR0_(i_elem, i_qp, 3 + i, 3 + j) = RR0(i, j);
            }
        }
    }
};
}  // namespace openturbine
