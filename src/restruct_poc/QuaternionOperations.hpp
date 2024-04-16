#pragma once

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

KOKKOS_INLINE_FUNCTION void QuaternionDerivative(const Quaternion& q, double m[3][4]) {
    auto q0 = q.GetScalarComponent();
    auto q1 = q.GetXComponent();
    auto q2 = q.GetYComponent();
    auto q3 = q.GetZComponent();
    m[0][0] = -q1;
    m[0][1] = q0;
    m[0][2] = -q3;
    m[0][3] = q2;
    m[1][0] = -q2;
    m[1][1] = q3;
    m[1][2] = q0;
    m[1][3] = -q1;
    m[2][0] = -q3;
    m[2][1] = -q2;
    m[2][2] = q1;
    m[2][3] = q0;
}

template <typename Q1, typename Q2, typename QN>
KOKKOS_INLINE_FUNCTION void QuaternionCompose(Q1 q1, Q2 q2, QN qn) {
    qn(0) = q1(0) * q2(0) - q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3);
    qn(1) = q1(0) * q2(1) + q1(1) * q2(0) + q1(2) * q2(3) - q1(3) * q2(2);
    qn(2) = q1(0) * q2(2) - q1(1) * q2(3) + q1(2) * q2(0) + q1(3) * q2(1);
    qn(3) = q1(0) * q2(3) + q1(1) * q2(2) - q1(2) * q2(1) + q1(3) * q2(0);
}

}
