#pragma once

#include <Kokkos_Core.hpp>

/**
 * @brief Computes AX(A) of a square matrix
 * @details AX(A) = tr(A)/2 * I - A/2, where I is the identity matrix
 * @param A Input square matrix
 * @param AX_A Output matrix containing the result
 */
template <typename Matrix>
KOKKOS_INLINE_FUNCTION void AX_Matrix(const Matrix& A, const Matrix& AX_A) {
    double trace{0.};
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

/**
 * @brief Computes the axial vector (also known as the vector representation) of a 3x3 skew-symmetric
 * matrix
 * @details The axial vector is defined as [w₁, w₂, w₃]ᵀ where:
 *          w₁ = (m₃₂ - m₂₃)/2
 *          w₂ = (m₁₃ - m₃₁)/2
 *          w₃ = (m₂₁ - m₁₂)/2
 * @param m Input 3x3 rotation matrix
 * @param v Output vector to store the result
 * @pre Matrix m must be 3x3 and vector v must have size 3
 */
template <typename Matrix, typename Vector>
KOKKOS_INLINE_FUNCTION void AxialVectorOfMatrix(const Matrix& m, const Vector& v) {
    v(0) = (m(2, 1) - m(1, 2)) / 2.;
    v(1) = (m(0, 2) - m(2, 0)) / 2.;
    v(2) = (m(1, 0) - m(0, 1)) / 2.;
}
