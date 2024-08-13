#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateRR0 {
    size_t i_elem;
    Kokkos::View<double** [4]>::const_type qp_r0_;
    Kokkos::View<double** [4]>::const_type qp_r_;
    Kokkos::View<double** [6][6]> qp_RR0_;

    KOKKOS_FUNCTION void operator()(const int i_qp) const {
        auto RR0_quaternion_data = Kokkos::Array<double, 4>{};
        auto RR0_quaternion = Kokkos::View<double[4]>(RR0_quaternion_data.data());
        QuaternionCompose(
            Kokkos::subview(qp_r_, i_elem, i_qp, Kokkos::ALL),
            Kokkos::subview(qp_r0_, i_elem, i_qp, Kokkos::ALL), RR0_quaternion
        );
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
