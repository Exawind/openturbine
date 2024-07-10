#pragma once

#include <array>

#include <Kokkos_Core.hpp>

namespace openturbine {

/// Converts a 4x1 quaternion to a 3x3 rotation matrix and returns the result
template <typename Quaternion, typename RotationMatrix>
KOKKOS_INLINE_FUNCTION void QuaternionToRotationMatrix(Quaternion q, RotationMatrix R) {
    R(0, 0) = q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3);
    R(0, 1) = 2. * (q(1) * q(2) - q(0) * q(3));
    R(0, 2) = 2. * (q(1) * q(3) + q(0) * q(2));
    R(1, 0) = 2. * (q(1) * q(2) + q(0) * q(3));
    R(1, 1) = q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3);
    R(1, 2) = 2. * (q(2) * q(3) - q(0) * q(1));
    R(2, 0) = 2. * (q(1) * q(3) - q(0) * q(2));
    R(2, 1) = 2. * (q(2) * q(3) + q(0) * q(1));
    R(2, 2) = q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
}

/// Converts a 4x1 quaternion to a 3x3 rotation matrix and returns the result
inline std::array<Array_3, 3> QuaternionToRotationMatrix(Array_4 q) {
    return std::array<Array_3, 3>{{
        {
            q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
            2. * (q[1] * q[2] - q[0] * q[3]),
            2. * (q[1] * q[3] + q[0] * q[2]),
        },
        {
            2. * (q[1] * q[2] + q[0] * q[3]),
            q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3],
            2. * (q[2] * q[3] - q[0] * q[1]),
        },
        {
            2. * (q[1] * q[3] - q[0] * q[2]),
            2. * (q[2] * q[3] + q[0] * q[1]),
            q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
        },
    }};
}

/// Rotates provided vector by provided *unit* quaternion and returns the result
template <typename Quaternion, typename View1, typename View2>
KOKKOS_INLINE_FUNCTION void RotateVectorByQuaternion(Quaternion q, View1 v, View2 v_rot) {
    v_rot(0) = (q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3)) * v(0) +
               2. * (q(1) * q(2) - q(0) * q(3)) * v(1) + 2. * (q(1) * q(3) + q(0) * q(2)) * v(2);
    v_rot(1) = 2. * (q(1) * q(2) + q(0) * q(3)) * v(0) +
               (q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3)) * v(1) +
               2. * (q(2) * q(3) - q(0) * q(1)) * v(2);
    v_rot(2) = 2. * (q(1) * q(3) - q(0) * q(2)) * v(0) + 2. * (q(2) * q(3) + q(0) * q(1)) * v(1) +
               (q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3)) * v(2);
}

inline std::array<double, 3> RotateVectorByQuaternion(
    std::array<double, 4> q, std::array<double, 3> v
) {
    auto v_rot = std::array<double, 3>{};
    v_rot[0] = (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]) * v[0] +
               2. * (q[1] * q[2] - q[0] * q[3]) * v[1] + 2. * (q[1] * q[3] + q[0] * q[2]) * v[2];
    v_rot[1] = 2. * (q[1] * q[2] + q[0] * q[3]) * v[0] +
               (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]) * v[1] +
               2. * (q[2] * q[3] - q[0] * q[1]) * v[2];
    v_rot[2] = 2. * (q[1] * q[3] - q[0] * q[2]) * v[0] + 2. * (q[2] * q[3] + q[0] * q[1]) * v[1] +
               (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]) * v[2];
    return v_rot;
}

/// Computes the derivative of a quaternion and stores the result in a 3x4 matrix
template <typename Quaternion, typename Matrix>
KOKKOS_INLINE_FUNCTION void QuaternionDerivative(Quaternion q, Matrix m) {
    m(0, 0) = -q(1);
    m(0, 1) = q(0);
    m(0, 2) = -q(3);
    m(0, 3) = q(2);
    m(1, 0) = -q(2);
    m(1, 1) = q(3);
    m(1, 2) = q(0);
    m(1, 3) = -q(1);
    m(2, 0) = -q(3);
    m(2, 1) = -q(2);
    m(2, 2) = q(1);
    m(2, 3) = q(0);
}

/// Computes the inverse of a quaternion
template <typename QuaternionInput, typename QuaternionOutput>
KOKKOS_INLINE_FUNCTION void QuaternionInverse(QuaternionInput q_in, QuaternionOutput q_out) {
    auto length =
        Kokkos::sqrt(q_in(0) * q_in(0) + q_in(1) * q_in(1) + q_in(2) * q_in(2) + q_in(3) * q_in(3));

    // Inverse of a quaternion is the conjugate divided by the length
    q_out(0) = q_in(0) / length;
    for (int i = 1; i < 4; ++i) {
        q_out(i) = -q_in(i) / length;
    }
}

/// Composes (i.e. multiplies) two quaternions and stores the result in a third quaternion
template <typename Quaternion1, typename Quaternion2, typename QuaternionN>
KOKKOS_INLINE_FUNCTION void QuaternionCompose(Quaternion1 q1, Quaternion2 q2, QuaternionN qn) {
    qn(0) = q1(0) * q2(0) - q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3);
    qn(1) = q1(0) * q2(1) + q1(1) * q2(0) + q1(2) * q2(3) - q1(3) * q2(2);
    qn(2) = q1(0) * q2(2) - q1(1) * q2(3) + q1(2) * q2(0) + q1(3) * q2(1);
    qn(3) = q1(0) * q2(3) + q1(1) * q2(2) - q1(2) * q2(1) + q1(3) * q2(0);
}

inline std::array<double, 4> QuaternionCompose(std::array<double, 4> q1, std::array<double, 4> q2) {
    auto qn = std::array<double, 4>{};
    qn[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    qn[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    qn[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    qn[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    return qn;
}

/// Computes the axial vector of a 3x3 rotation matrix
template <typename Matrix, typename Vector>
KOKKOS_INLINE_FUNCTION void AxialVectorOfMatrix(Matrix m, Vector v) {
    v(0) = (m(2, 1) - m(1, 2)) / 2.;
    v(1) = (m(0, 2) - m(2, 0)) / 2.;
    v(2) = (m(1, 0) - m(0, 1)) / 2.;
}

/// Computes AX(A) of a square matrix
template <typename Matrix>
KOKKOS_INLINE_FUNCTION void AX_Matrix(Matrix A, Matrix AX_A) {
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

/// Returns a 4-D quaternion from provided 3-D rotation vector, i.e. the exponential map
template <typename Vector, typename Quaternion>
KOKKOS_INLINE_FUNCTION void RotationVectorToQuaternion(Vector phi, Quaternion quaternion) {
    const auto angle = Kokkos::sqrt(phi(0) * phi(0) + phi(1) * phi(1) + phi(2) * phi(2));
    const auto cos_angle = Kokkos::cos(angle / 2.);
    const auto factor = (Kokkos::abs(angle) < 1.e-12) ? 0. : Kokkos::sin(angle / 2.) / angle;

    quaternion(0) = cos_angle;
    for (int i = 1; i < 4; ++i) {
        quaternion(i) = phi(i - 1) * factor;
    }
}

inline std::array<double, 4> RotationVectorToQuaternion(std::array<double, 3> phi) {
    const auto angle = std::sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);

    if (std::abs(angle) < 1e-12) {
        return std::array<double, 4>{1., 0., 0., 0.};
    }

    const auto sin_angle = std::sin(angle / 2.);
    const auto cos_angle = std::cos(angle / 2.);
    const auto factor = sin_angle / angle;
    return std::array<double, 4>{cos_angle, phi[0] * factor, phi[1] * factor, phi[2] * factor};
}

}  // namespace openturbine
