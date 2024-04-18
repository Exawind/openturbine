#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

struct CalculateTemporaryVariables {
    View_Nx3::const_type qp_x0_prime_;
    View_Nx3::const_type qp_u_prime_;
    View_Nx3x3 x0pupSS_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        Vector x0_prime(qp_x0_prime_(i_qp, 0), qp_x0_prime_(i_qp, 1), qp_x0_prime_(i_qp, 2));
        Vector u_prime(qp_u_prime_(i_qp, 0), qp_u_prime_(i_qp, 1), qp_u_prime_(i_qp, 2));
        auto x0pup = x0_prime + u_prime;
        double tmp[3][3];
        x0pup.Tilde(tmp);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                x0pupSS_(i_qp, i, j) = tmp[i][j];
            }
        }
    }
};

}  // namespace openturbine
