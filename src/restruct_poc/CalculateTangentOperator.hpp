#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "VectorOperations.hpp"
#include "MatrixOperations.hpp"

namespace openturbine {

struct CalculateTangentOperator {
    double h;
    View_Nx6::const_type q_delta;
    View_NxN T;

    KOKKOS_FUNCTION
    void operator()(const int i_node) const {
        int j = i_node * kLieAlgebraComponents;
        for (int k = 0; k < kLieAlgebraComponents; ++k) {
            T(j + k, j + k) = 1.0;
        }
        double rv[3] = {h * q_delta(i_node, 3), h * q_delta(i_node, 4), h * q_delta(i_node, 5)};
        double phi = Kokkos::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
        if (phi > 1.0e-16) {
            j += 3;
            double m1[3][3], m2[3][3], m3[3][3], m4[3][3];
            double tmp1 = (Kokkos::cos(phi) - 1.) / (phi * phi);
            double tmp2 = (1. - Kokkos::sin(phi) / phi) / (phi * phi);
            VectorTilde(tmp1, rv, m1);
            VectorTilde(tmp2, rv, m2);
            VectorTilde(1.0, rv, m3);
            Mat3xMat3(m2, m3, m4);
            for (int k = 0; k < 3; ++k) {
                for (int n = 0; n < 3; ++n) {
                    T(j + k, j + n) += m1[k][n] + m4[k][n];
                }
            }
        }
    }
};

}
