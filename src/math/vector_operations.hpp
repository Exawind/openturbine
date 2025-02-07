#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

/// Converts a 3x1 vector to a 3x3 skew-symmetric matrix and returns the result
template <typename VectorType, typename MatrixType>
KOKKOS_INLINE_FUNCTION void VecTilde(const VectorType& vector, const MatrixType& matrix) {
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
KOKKOS_INLINE_FUNCTION double DotProduct(const VectorType& a, const VectorType& b) {
    double sum = 0.;
    for (int i = 0; i < a.extent_int(0); ++i) {
        sum += a(i) * b(i);
    }
    return sum;
}

/// Calculate the dot product between two vector views
constexpr double DotProduct(const Array_3& a, const Array_3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Calculate the cross product between two vector views
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void CrossProduct(
    const VectorType& a, const VectorType& b, const VectorType& c
) {
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

/// Calculate the norm of a given vector
constexpr double Norm(const Array_3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/// UnitVector returns the unit vector of the given vector
constexpr Array_3 UnitVector(const Array_3& v) {
    const double norm = Norm(v);
    if (norm == 0.) {
        throw std::invalid_argument("Cannot normalize a zero vector");
    }
    return Array_3{
        v[0] / norm,
        v[1] / norm,
        v[2] / norm,
    };
}

/// Dot3 returns the dot product of two 3-component vectors
constexpr double Dot3(const Array_3& v1, const Array_3& v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

}  // namespace openturbine
