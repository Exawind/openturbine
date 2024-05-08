#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

/// Converts a 3x1 vector to a 3x3 skew-symmetric matrix and returns the result
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

constexpr std::array<double, 3> CrossProduct(std::array<double, 3> a, std::array<double, 3> b) {
    auto c = std::array<double, 3>{};
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

}  // namespace openturbine
