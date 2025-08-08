#pragma once

#include <array>

#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine::math {

/**
 * @brief Computes AX(A) of a square matrix
 * @details AX(A) = tr(A)/2 * I - A/2, where I is the identity matrix
 * @param A Input square matrix
 * @param AX_A Output matrix containing the result
 */
template <typename Matrix>
KOKKOS_INLINE_FUNCTION void AX_Matrix(const Matrix& A, const Matrix& AX_A) {
    double trace{0.};
    for (auto i = 0; i < A.extent_int(0); ++i) {
        trace += A(i, i);
    }

    trace /= 2.;

    for (auto i = 0; i < A.extent_int(0); ++i) {
        for (auto j = 0; j < A.extent_int(1); ++j) {
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

inline std::array<std::array<double, 6>, 6> RotateMatrix6(
    const std::array<std::array<double, 6>, 6>& m, const std::array<double, 4>& q
) {
    const auto rm = QuaternionToRotationMatrix(q);
    std::array<std::array<double, 6>, 6> r{{
        {rm[0][0], rm[0][1], rm[0][2], 0., 0., 0.},
        {rm[1][0], rm[1][1], rm[1][2], 0., 0., 0.},
        {rm[2][0], rm[2][1], rm[2][2], 0., 0., 0.},
        {0., 0., 0., rm[0][0], rm[0][1], rm[0][2]},
        {0., 0., 0., rm[1][0], rm[1][1], rm[1][2]},
        {0., 0., 0., rm[2][0], rm[2][1], rm[2][2]},
    }};

    // matmul(r,m)
    std::array<std::array<double, 6>, 6> mt{};
    for (auto i = 0U; i < 6; ++i) {
        for (auto j = 0U; j < 6; ++j) {
            mt[i][j] = 0.;
            for (auto k = 0U; k < 6; ++k) {
                mt[i][j] += r[i][k] * m[k][j];
            }
        }
    }

    // matmul(matmul(r,m),r^T)
    std::array<std::array<double, 6>, 6> mo{};
    for (auto i = 0U; i < 6; ++i) {
        for (auto j = 0U; j < 6; ++j) {
            mo[i][j] = 0.;
            for (auto k = 0U; k < 6; ++k) {
                mo[i][j] += mt[i][k] * r[j][k];
            }
        }
    }

    return mo;
}

}  // namespace openturbine::math
