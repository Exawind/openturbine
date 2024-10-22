#pragma once

#include <Kokkos_Core.hpp>

/// Computes AX(A) of a square matrix
template <typename Matrix>
KOKKOS_INLINE_FUNCTION void AX_Matrix(const Matrix& A, const Matrix& AX_A) {
    double trace = 0.;
    for (int i = 0; i < A.extent_int(0); ++i) {
        trace += A(i, i);
    }
    trace /= 2.;
    for (int i = 0; i < A.extent_int(0); ++i) {
        for (int j = 0; j < A.extent_int(1); ++j) {
            AX_A(i, j) = -A(i, j) / 2.;
        }
        AX_A(i, i) += trace;
    }
}

/// Computes the axial vector of a 3x3 rotation matrix
template <typename Matrix, typename Vector>
KOKKOS_INLINE_FUNCTION void AxialVectorOfMatrix(const Matrix& m, const Vector& v) {
    v(0) = (m(2, 1) - m(1, 2)) / 2.;
    v(1) = (m(0, 2) - m(2, 0)) / 2.;
    v(2) = (m(1, 0) - m(0, 1)) / 2.;
}
