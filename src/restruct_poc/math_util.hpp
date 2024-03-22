#pragma once

#include "types.hpp"

namespace oturb {

KOKKOS_INLINE_FUNCTION
void InterpVector3(View_NxN shape_matrix, View_Nx3 node_v, View_Nx3 qp_v) {
    Kokkos::deep_copy(qp_v, 0.);
    for (size_t i = 0; i < node_v.extent(0); ++i) {
        for (size_t j = 0; j < qp_v.extent(0); ++j) {
            for (size_t k = 0; k < 3; ++k) {
                qp_v(j, k) += node_v(i, k) * shape_matrix(i, j);
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector4(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v) {
    Kokkos::deep_copy(qp_v, 0.);
    for (size_t i = 0; i < node_v.extent(0); ++i) {
        for (size_t j = 0; j < qp_v.extent(0); ++j) {
            for (size_t k = 0; k < 4; ++k) {
                qp_v(j, k) += node_v(i, k) * shape_matrix(i, j);
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpQuaternion(View_NxN shape_matrix, View_Nx4 node_v, View_Nx4 qp_v) {
    InterpVector4(shape_matrix, node_v, qp_v);

    // Normalize quaternions (rows)
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        auto length = Kokkos::sqrt(
            Kokkos::pow(qp_v(j, 0), 2) + Kokkos::pow(qp_v(j, 1), 2) + Kokkos::pow(qp_v(j, 2), 2) +
            Kokkos::pow(qp_v(j, 3), 2)
        );
        if (length == 0.) {
            qp_v(j, 0) = 1.;
            qp_v(j, 1) = 0.;
            qp_v(j, 2) = 0.;
            qp_v(j, 3) = 0.;
        } else {
            qp_v(j, 0) /= length;
            qp_v(j, 1) /= length;
            qp_v(j, 2) /= length;
            qp_v(j, 3) /= length;
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector3Deriv(
    View_NxN shape_matrix_deriv, View_N jacobian, View_Nx3 node_v, View_Nx3 qp_v
) {
    InterpVector3(shape_matrix_deriv, node_v, qp_v);
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        for (size_t k = 0; k < qp_v.extent(1); ++k) {
            qp_v(j, k) /= jacobian(j);
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector4Deriv(
    View_NxN shape_matrix_deriv, View_N jacobian, View_Nx4 node_v, View_Nx4 qp_v
) {
    InterpVector4(shape_matrix_deriv, node_v, qp_v);
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        for (size_t k = 0; k < qp_v.extent(1); ++k) {
            qp_v(j, k) /= jacobian(j);
        }
    }
}

//------------------------------------------------------------------------------
// Vector
//------------------------------------------------------------------------------

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulAB(View_A A, View_B B, View_C C) {
    Kokkos::deep_copy(C, 0.);
    for (size_t i = 0; i < A.extent(0); ++i) {
        for (size_t k = 0; k < B.extent(0); ++k) {
            C(i) += A(i, k) * B(k);
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulATB(View_A A, View_B B, View_C C) {
    Kokkos::deep_copy(C, 0.);
    for (size_t i = 0; i < A.extent(1); ++i) {
        for (size_t k = 0; k < B.extent(0); ++k) {
            C(i) += A(k, i) * B(k);
        }
    }
}

template <typename View_3x3>
KOKKOS_INLINE_FUNCTION void VecTilde(View_3 vector, View_3x3 matrix) {
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

template <typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void VecScale(V1 v_in, double scale, V2 v_out) {
    for (size_t i = 0; i < v_in.extent(0); ++i) {
        v_out(i) = v_in(i) * scale;
    }
}

//------------------------------------------------------------------------------
// Matrix
//------------------------------------------------------------------------------

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION void MatScale(A m_in, double scale, B m_out) {
    for (size_t i = 0; i < m_in.extent(0); ++i) {
        for (size_t j = 0; j < m_in.extent(1); ++j) {
            m_out(i) = m_in(i) * scale;
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatAdd(View_A M_A, View_B M_B, View_C M_C) {
    for (size_t i = 0; i < M_A.extent(0); ++i) {
        for (size_t j = 0; j < M_A.extent(1); ++j) {
            M_C(i, j) = M_A(i, j) + M_B(i, j);
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulAB(View_A A, View_B B, View_C C) {
    Kokkos::deep_copy(C, 0.);
    for (size_t i = 0; i < A.extent(0); ++i) {
        for (size_t j = 0; j < B.extent(1); ++j) {
            for (size_t k = 0; k < B.extent(0); ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulATB(View_A AT, View_B B, View_C C) {
    Kokkos::deep_copy(C, 0.);
    for (size_t i = 0; i < AT.extent(1); ++i) {
        for (size_t j = 0; j < B.extent(1); ++j) {
            for (size_t k = 0; k < B.extent(0); ++k) {
                C(i, j) += AT(k, i) * B(k, j);
            }
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulABT(View_A A, View_B BT, View_C C) {
    Kokkos::deep_copy(C, 0.);
    for (size_t i = 0; i < A.extent(0); ++i) {
        for (size_t j = 0; j < BT.extent(0); ++j) {
            for (size_t k = 0; k < BT.extent(1); ++k) {
                C(i, j) += A(i, k) * BT(j, k);
            }
        }
    }
}

/// Populates a 3x3 rotation matrix from a 4x1 quaternion
template <typename View_Rot>
KOKKOS_INLINE_FUNCTION void QuaternionToRotationMatrix(View_Quat q, View_Rot R) {
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

/// Rotates provided vector by provided *unit* quaternion
KOKKOS_INLINE_FUNCTION
void QuaternionRotateVector(View_Quat q, View_3 v, View_3 v_rot) {
    v_rot[0] = (q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3)) * v(0) +
               2. * (q(1) * q(2) - q(0) * q(3)) * v(1) + 2. * (q(1) * q(3) + q(0) * q(2)) * v(2);
    v_rot[1] = 2. * (q(1) * q(2) + q(0) * q(3)) * v(0) +
               (q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3)) * v(1) +
               2. * (q(2) * q(3) - q(0) * q(1)) * v(2);
    v_rot[2] = 2. * (q(1) * q(3) - q(0) * q(2)) * v(0) + 2. * (q(2) * q(3) + q(0) * q(1)) * v(1) +
               (q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3)) * v(2);
}

/// Calculate the quaternion derivative (E)
KOKKOS_INLINE_FUNCTION
void QuaternionDerivative(View_Quat q, View_3x4 m) {
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

/// Multiplies provided quaternion with this quaternion and returns the result
KOKKOS_INLINE_FUNCTION
void QuaternionCompose(View_Quat q1, View_Quat q2, View_Quat qn) {
    qn(0) = q1(0) * q2(0) - q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3);
    qn(1) = q1(0) * q2(1) + q1(1) * q2(0) + q1(2) * q2(3) - q1(3) * q2(2);
    qn(2) = q1(0) * q2(2) - q1(1) * q2(3) + q1(2) * q2(0) + q1(3) * q2(1);
    qn(3) = q1(0) * q2(3) + q1(1) * q2(2) - q1(2) * q2(1) + q1(3) * q2(0);
}

}  // namespace oturb