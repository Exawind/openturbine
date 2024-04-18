#pragma once

#include <Kokkos_Core.hpp>

#include "MatrixOperations.hpp"
#include "QuaternionOperations.hpp"
#include "types.hpp"

namespace openturbine {

struct CalculateStrain {
    View_Nx3::const_type qp_x0_prime_;  //
    View_Nx3::const_type qp_u_prime_;   //
    View_Nx4::const_type qp_r_;         //
    View_Nx4::const_type qp_r_prime_;   //
    View_Nx6 qp_strain_;                //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        Vector x0_prime(qp_x0_prime_(i_qp, 0), qp_x0_prime_(i_qp, 1), qp_x0_prime_(i_qp, 2));
        Vector u_prime(qp_u_prime_(i_qp, 0), qp_u_prime_(i_qp, 1), qp_u_prime_(i_qp, 2));
        Quaternion R(qp_r_(i_qp, 0), qp_r_(i_qp, 1), qp_r_(i_qp, 2), qp_r_(i_qp, 3));

        auto R_x0_prime = R * x0_prime;
        auto e1 = x0_prime + u_prime - R_x0_prime;

        double E[3][4];
        QuaternionDerivative(R, E);
        double R_prime[4] = {
            qp_r_prime_(i_qp, 0), qp_r_prime_(i_qp, 1), qp_r_prime_(i_qp, 2), qp_r_prime_(i_qp, 3)};
        double e2[3];

        for (int i = 0; i < 3; ++i) {
            e2[i] = 0.;
            for (int k = 0; k < 4; ++k) {
                e2[i] += E[i][k] * R_prime[k];
            }
        }

        qp_strain_(i_qp, 0) = e1.GetX();
        qp_strain_(i_qp, 1) = e1.GetY();
        qp_strain_(i_qp, 2) = e1.GetZ();
        qp_strain_(i_qp, 3) = 2.0 * e2[0];
        qp_strain_(i_qp, 4) = 2.0 * e2[1];
        qp_strain_(i_qp, 5) = 2.0 * e2[2];
    }
};

}  // namespace openturbine
