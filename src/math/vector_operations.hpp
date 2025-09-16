#pragma once

#include <array>

#include <Kokkos_Core.hpp>

namespace openturbine::math {

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
template <typename AVectorType, typename BVectorType>
KOKKOS_INLINE_FUNCTION double DotProduct(const AVectorType& a, const BVectorType& b) {
    double sum = 0.;
    for (auto i = 0; i < a.extent_int(0); ++i) {
        sum += a(i) * b(i);
    }
    return sum;
}

/// Calculate the dot product between two vector views
constexpr double DotProduct(const std::array<double, 3>& a, const std::array<double, 3>& b) {
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
constexpr std::array<double, 3> CrossProduct(
    std::span<const double, 3> a, std::span<const double, 3> b
) {
    return std::array<double, 3>{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

/// Calculate the norm of a given vector
constexpr double Norm(const std::array<double, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/// UnitVector returns the unit vector of the given vector
constexpr std::array<double, 3> UnitVector(const std::array<double, 3>& v) {
    const double norm = Norm(v);
    if (norm == 0.) {
        throw std::invalid_argument("Cannot normalize a zero vector");
    }
    return std::array<double, 3>{
        v[0] / norm,
        v[1] / norm,
        v[2] / norm,
    };
}
}  // namespace openturbine::math
