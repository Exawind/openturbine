#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "src/gebt_poc/quadrature.h"

namespace openturbine {

struct CalculateRR0 {
    View_Nx4::const_type qp_r0_;  // quadrature point initial rotation
    View_Nx4::const_type qp_r_;   // quadrature rotation displacement
    View_Nx6x6 qp_RR0_;           // quadrature global rotation

    KOKKOS_FUNCTION void operator()(const int i_qp) const {
        Quaternion R(qp_r_(i_qp, 0), qp_r_(i_qp, 1), qp_r_(i_qp, 2), qp_r_(i_qp, 3));
        Quaternion R0(qp_r0_(i_qp, 0), qp_r0_(i_qp, 1), qp_r0_(i_qp, 2), qp_r0_(i_qp, 3));
        auto RR0 = (R * R0).to_rotation_matrix();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                qp_RR0_(i_qp, i, j) = RR0(i, j);
                qp_RR0_(i_qp, 3 + i, 3 + j) = RR0(i, j);
            }
        }
    }
};
}
