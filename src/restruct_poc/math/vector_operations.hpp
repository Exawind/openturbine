#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

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

/// Calculate the dot product between two vector views
template <typename VectorType>
KOKKOS_INLINE_FUNCTION double DotProduct(VectorType a, VectorType b) {
    double sum = 0.;
    for (int i = 0; i < a.extent_int(0); ++i) {
        sum += a(i) * b(i);
    }
    return sum;
}

/// Calculate the dot product between two vector views
constexpr double DotProduct(Array_3 a, Array_3 b) {
    double sum = 0.;
    for (int i = 0; i < 3; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/// Calculate the cross product between two vector views
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void CrossProduct(VectorType a, VectorType b, VectorType c) {
    c(0) = a(1) * b(2) - a(2) * b(1);
    c(1) = a(2) * b(0) - a(0) * b(2);
    c(2) = a(0) * b(1) - a(1) * b(0);
}

/// Calculate the cross product between two vectors
constexpr Array_3 CrossProduct(const Array_3& a, const Array_3& b) {
    return Array_3{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

/// UnitVector returns the unit vector of the given vector
constexpr Array_3 UnitVector(const Array_3& v) {
    // Calculate vector norm
    double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (norm == 0.) {
        throw std::invalid_argument("Cannot normalize a zero vector");
    }
    return Array_3{
        v[0] / norm,
        v[1] / norm,
        v[2] / norm,
    };
}

}  // namespace openturbine
