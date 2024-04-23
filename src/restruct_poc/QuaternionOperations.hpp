#pragma once
#include <array>
<<<<<<< HEAD
=======

>>>>>>> fdc3be9 (Remove Vector and Quaternion includes from types.)
#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename Q, typename View_Rotation>
KOKKOS_INLINE_FUNCTION void QuaternionToRotationMatrix(Q q, View_Rotation R) {
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

template <typename Q, typename View1, typename View2>
KOKKOS_INLINE_FUNCTION void QuaternionRotateVector(Q q, View1 v, View2 v_rot) {
    v_rot[0] = (q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3)) * v(0) +
               2. * (q(1) * q(2) - q(0) * q(3)) * v(1) + 2. * (q(1) * q(3) + q(0) * q(2)) * v(2);
    v_rot[1] = 2. * (q(1) * q(2) + q(0) * q(3)) * v(0) +
               (q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3)) * v(1) +
               2. * (q(2) * q(3) - q(0) * q(1)) * v(2);
    v_rot[2] = 2. * (q(1) * q(3) - q(0) * q(2)) * v(0) + 2. * (q(2) * q(3) + q(0) * q(1)) * v(1) +
               (q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3)) * v(2);
}

inline std::array<double, 3> QuaternionRotateVector(
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

template <typename Q, typename M>
KOKKOS_INLINE_FUNCTION void QuaternionDerivative(Q q, M m) {
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

template <typename Qin, typename Qout>
KOKKOS_INLINE_FUNCTION void QuaternionInverse(Qin q_in, Qout q_out) {
    auto length =
        Kokkos::sqrt(q_in(0) * q_in(0) + q_in(1) * q_in(1) + q_in(2) * q_in(2) + q_in(3) * q_in(3));
    q_out(0) = q_in(0) / length;
    for (int i = 1; i < 4; ++i) {
        q_out(i) = -q_in(i) / length;
    }
}

template <typename Q1, typename Q2, typename QN>
KOKKOS_INLINE_FUNCTION void QuaternionCompose(Q1 q1, Q2 q2, QN qn) {
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

template <typename M, typename V>
KOKKOS_INLINE_FUNCTION void ComputeAxialVector(M m, V v) {
    v(0) = (m(2, 1) - m(1, 2)) / 2.;
    v(1) = (m(0, 2) - m(2, 0)) / 2.;
    v(2) = (m(1, 0) - m(0, 1)) / 2.;
}

template <typename V, typename Q>
KOKKOS_INLINE_FUNCTION void RotationVectorToQuaternion(V phi, Q quaternion) {
    const auto angle = Kokkos::sqrt(phi(0) * phi(0) + phi(1) * phi(1) + phi(2) * phi(2));
    const auto cos_angle = Kokkos::cos(angle / 2.0);
    const auto factor = (Kokkos::abs(angle) < 1.e-12) ? 0. : Kokkos::sin(angle / 2.0) / angle;
    quaternion(0) = cos_angle;
    for (int i = 0; i < 3; ++i) {
        quaternion(i + 1) = phi(i) * factor;
    }
}

inline std::array<double, 4> RotationVectorToQuaternion(std::array<double, 3> phi) {
    const auto angle = std::sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);

    if (std::abs(angle) < 1e-12) {
        return std::array<double, 4>{1., 0., 0., 0.};
    } else {
        const auto sin_angle = std::sin(angle / 2.0);
        const auto cos_angle = std::cos(angle / 2.0);
        const auto factor = sin_angle / angle;
        return std::array<double, 4>{cos_angle, phi[0] * factor, phi[1] * factor, phi[2] * factor};
    }
}

}  // namespace openturbine
