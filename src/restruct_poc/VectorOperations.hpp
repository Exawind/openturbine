#pragma once
#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename VectorType, typename MatrixType>
KOKKOS_INLINE_FUNCTION void VecTilde(VectorType vector, MatrixType matrix) {
    matrix(0, 0) = 0.;
    matrix(0, 1) = -vector(2);
    matrix(0, 2) = vector(1);
    matrix(1, 0) = vector(2);
    matrix(1, 1) = 0.;
    matrix(1, 2) = -vector(0);
    matrix(2, 0) = -vector(1);
    matrix(2, 1) = vector(0);
    matrix(2, 2) = 0.;
}

KOKKOS_INLINE_FUNCTION
void VectorTilde(double a, double v[3], double m[3][3]) {
    m[0][0] = 0.;
    m[0][1] = -v[2] * a;
    m[0][2] = v[1] * a;
    m[1][0] = v[2] * a;
    m[1][1] = 0.;
    m[1][2] = -v[0] * a;
    m[2][0] = -v[1] * a;
    m[2][1] = v[0] * a;
    m[2][2] = 0.;
}

}  // namespace openturbine
