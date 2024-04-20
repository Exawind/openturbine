#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

KOKKOS_INLINE_FUNCTION
void Mat3xMat3(double m1[3][3], double m2[3][3], double m3[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m3[i][j] = 0.;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                m3[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

}  // namespace openturbine
