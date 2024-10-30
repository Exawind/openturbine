#pragma once

#include <array>

#include <Kokkos_Core.hpp>

#include "src/types.hpp"

namespace openturbine {

/// Converts a 4x1 quaternion to a 3x3 rotation matrix and returns the result
template <typename Quaternion, typename RotationMatrix>
KOKKOS_INLINE_FUNCTION void QuaternionToRotationMatrix(
    const Quaternion& q, const RotationMatrix& R
) {
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
inline std::array<Array_3, 3> QuaternionToRotationMatrix(const Array_4& q) {
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
KOKKOS_INLINE_FUNCTION void RotateVectorByQuaternion(
    const Quaternion& q, const View1& v, const View2& v_rot
) {
    v_rot(0) = (q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3)) * v(0) +
               2. * (q(1) * q(2) - q(0) * q(3)) * v(1) + 2. * (q(1) * q(3) + q(0) * q(2)) * v(2);
    v_rot(1) = 2. * (q(1) * q(2) + q(0) * q(3)) * v(0) +
               (q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3)) * v(1) +
               2. * (q(2) * q(3) - q(0) * q(1)) * v(2);
    v_rot(2) = 2. * (q(1) * q(3) - q(0) * q(2)) * v(0) + 2. * (q(2) * q(3) + q(0) * q(1)) * v(1) +
               (q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3)) * v(2);
}

inline std::array<double, 3> RotateVectorByQuaternion(const Array_4& q, const Array_3& v) {
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
KOKKOS_INLINE_FUNCTION void QuaternionDerivative(const Quaternion& q, const Matrix& m) {
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
KOKKOS_INLINE_FUNCTION void QuaternionInverse(
    const QuaternionInput& q_in, const QuaternionOutput& q_out
) {
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
KOKKOS_INLINE_FUNCTION void QuaternionCompose(
    const Quaternion1& q1, const Quaternion2& q2, QuaternionN& qn
) {
    qn(0) = q1(0) * q2(0) - q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3);
    qn(1) = q1(0) * q2(1) + q1(1) * q2(0) + q1(2) * q2(3) - q1(3) * q2(2);
    qn(2) = q1(0) * q2(2) - q1(1) * q2(3) + q1(2) * q2(0) + q1(3) * q2(1);
    qn(3) = q1(0) * q2(3) + q1(1) * q2(2) - q1(2) * q2(1) + q1(3) * q2(0);
}

inline Array_4 QuaternionCompose(const Array_4& q1, const Array_4& q2) {
    auto qn = std::array<double, 4>{};
    qn[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    qn[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    qn[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    qn[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    return qn;
}

/// Returns a 4-D quaternion from provided 3-D rotation vector, i.e. the exponential map
template <typename Vector, typename Quaternion>
KOKKOS_INLINE_FUNCTION void RotationVectorToQuaternion(
    const Vector& phi, const Quaternion& quaternion
) {
    const auto angle = Kokkos::sqrt(phi(0) * phi(0) + phi(1) * phi(1) + phi(2) * phi(2));
    const auto cos_angle = Kokkos::cos(angle / 2.);
    const auto factor = (Kokkos::abs(angle) < 1.e-12) ? 0. : Kokkos::sin(angle / 2.) / angle;

    quaternion(0) = cos_angle;
    for (int i = 1; i < 4; ++i) {
        quaternion(i) = phi(i - 1) * factor;
    }
}

/// Returns a 3-D rotation vector from provided 4-D quaternion
template <typename Quaternion, typename Vector>
KOKKOS_INLINE_FUNCTION void QuaternionToRotationVector(
    const Quaternion& quaternion, const Vector& phi
) {
    auto theta = 2. * Kokkos::acos(quaternion(0));
    const auto sin_half_theta = std::sqrt(1. - quaternion(0) * quaternion(0));
    if (sin_half_theta > 1e-12) {
        phi(0) = theta * quaternion(1) / sin_half_theta;
        phi(1) = theta * quaternion(2) / sin_half_theta;
        phi(2) = theta * quaternion(3) / sin_half_theta;
    } else {
        phi(0) = 0.;
        phi(1) = 0.;
        phi(2) = 0.;
    }
}

/// Returns a 3-D rotation vector from provided 4-D quaternion
inline Array_3 QuaternionToRotationVector(const Array_4& quaternion) {
    auto theta = 2. * Kokkos::acos(quaternion[0]);
    const auto sin_half_theta = std::sqrt(1. - quaternion[0] * quaternion[0]);
    Array_3 phi;
    if (sin_half_theta > 1e-12) {
        phi[0] = theta * quaternion[1] / sin_half_theta;
        phi[1] = theta * quaternion[2] / sin_half_theta;
        phi[2] = theta * quaternion[3] / sin_half_theta;
    } else {
        phi[0] = 0.;
        phi[1] = 0.;
        phi[2] = 0.;
    }
    return phi;
}

inline Array_4 RotationVectorToQuaternion(const Array_3& phi) {
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
