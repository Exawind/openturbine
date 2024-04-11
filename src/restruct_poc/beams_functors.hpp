#pragma once

#include <array>
#include <numeric>

#include <KokkosBlas.hpp>

#include "beams_data.hpp"

#include "src/gebt_poc/quadrature.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine {

//------------------------------------------------------------------------------
// Vector Functions
//------------------------------------------------------------------------------

template <typename View_A>
KOKKOS_INLINE_FUNCTION void FillVector(View_A A, double value) {
    for (size_t i = 0; i < A.extent(0); ++i) {
        A(i) = value;
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulAB(View_A A, View_B B, View_C C) {
    for (size_t i = 0; i < A.extent(0); ++i) {
        C(i) = 0.;
        for (size_t k = 0; k < B.extent(0); ++k) {
            C(i) += A(i, k) * B(k);
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulATB(View_A A, View_B B, View_C C) {
    for (size_t i = 0; i < A.extent(1); ++i) {
        C(i) = 0.;
        for (size_t k = 0; k < B.extent(0); ++k) {
            C(i) += A(k, i) * B(k);
        }
    }
}

template <typename VectorType, typename MatrixType>
KOKKOS_INLINE_FUNCTION void VecTilde(VectorType vector, MatrixType matrix) {
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
// Matrix functions
//------------------------------------------------------------------------------

template <typename View_A>
KOKKOS_INLINE_FUNCTION void FillMatrix(View_A A, double value) {
    for (size_t i = 0; i < A.extent(0); ++i) {
        for (size_t j = 0; j < A.extent(1); ++j) {
            A(i, j) = value;
        }
    }
}

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
    for (size_t i = 0; i < A.extent(0); ++i) {
        for (size_t j = 0; j < B.extent(1); ++j) {
            C(i, j) = 0.;
            for (size_t k = 0; k < B.extent(0); ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulATB(View_A AT, View_B B, View_C C) {
    for (size_t i = 0; i < AT.extent(1); ++i) {
        for (size_t j = 0; j < B.extent(1); ++j) {
            C(i, j) = 0.;
            for (size_t k = 0; k < B.extent(0); ++k) {
                C(i, j) += AT(k, i) * B(k, j);
            }
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulABT(View_A A, View_B BT, View_C C) {
    for (size_t i = 0; i < A.extent(0); ++i) {
        for (size_t j = 0; j < BT.extent(0); ++j) {
            C(i, j) = 0.;
            for (size_t k = 0; k < BT.extent(1); ++k) {
                C(i, j) += A(i, k) * BT(j, k);
            }
        }
    }
}

/// Populates a 3x3 rotation matrix from a 4x1 quaternion
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

/// Rotates provided vector by provided *unit* quaternion
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

/// Calculate the quaternion derivative (E)
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

/// Multiplies provided quaternion with this quaternion and returns the result
template <typename Q1, typename Q2, typename QN>
KOKKOS_INLINE_FUNCTION void QuaternionCompose(Q1 q1, Q2 q2, QN qn) {
    qn(0) = q1(0) * q2(0) - q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3);
    qn(1) = q1(0) * q2(1) + q1(1) * q2(0) + q1(2) * q2(3) - q1(3) * q2(2);
    qn(2) = q1(0) * q2(2) - q1(1) * q2(3) + q1(2) * q2(0) + q1(3) * q2(1);
    qn(3) = q1(0) * q2(3) + q1(1) * q2(2) - q1(2) * q2(1) + q1(3) * q2(0);
}

//------------------------------------------------------------------------------
// Interpolation functions
//------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void InterpVector3(View_NxN shape_matrix, View_Nx3 node_v, View_Nx3 qp_v) {
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        for (size_t k = 0; k < 3; ++k) {
            qp_v(j, k) = 0.;
        }
    }
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
    for (size_t j = 0; j < qp_v.extent(0); ++j) {
        for (size_t k = 0; k < 4; ++k) {
            qp_v(j, k) = 0.;
        }
    }
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
// Functors to perform calculations on Beams structure
//------------------------------------------------------------------------------

struct InterpolateQPPosition {
    Kokkos::View<Beams::ElemIndices*> elem_indices;  // Element indices
    View_NxN shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx7 node_pos_rot_;                          // Node global position vector
    View_Nx3 qp_pos_;                                // quadrature point position

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_pos =
            Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair((size_t)0, (size_t)3));
        auto qp_pos = Kokkos::subview(qp_pos_, idx.qp_range, Kokkos::ALL);

        // Initialize qp_pos
        for (size_t i = 0; i < qp_pos.extent(0); ++i) {
            for (size_t j = 0; j < qp_pos.extent(1); ++j) {
                qp_pos(i, j) = 0.;
            }
        }

        // Perform matrix-matrix multiplication
        for (size_t i = 0; i < idx.num_nodes; ++i) {
            for (size_t j = 0; j < idx.num_qps; ++j) {
                for (size_t k = 0; k < kVectorComponents; ++k) {
                    qp_pos(j, k) += node_pos(i, k) * shape_interp(i, j);
                }
            }
        }
    }
};

struct InterpolateQPRotation {
    Kokkos::View<Beams::ElemIndices*> elem_indices;  // Element indices
    View_NxN shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx7 node_pos_rot_;                          // Node global position vector
    View_Nx4 qp_rot_;                                // quadrature point rotation

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_rot = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_rot = Kokkos::subview(qp_rot_, idx.qp_range, Kokkos::ALL);

        InterpQuaternion(shape_interp, node_rot, qp_rot);
    }
};

struct InterpolateQPRotationDerivative {
    Kokkos::View<Beams::ElemIndices*> elem_indices;  // Element indices
    View_NxN shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_N qp_jacobian_;                             // Jacobians
    View_Nx7 node_pos_rot_;                          // Node global position/rotation vector
    View_Nx4 qp_rot_deriv_;                          // quadrature point rotation derivative

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_rot_deriv = Kokkos::subview(qp_rot_deriv_, idx.qp_range, Kokkos::ALL);
        auto node_rot = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        InterpVector4Deriv(shape_deriv, qp_jacobian, node_rot, qp_rot_deriv);
    }
};

struct CalculateJacobian {
    Kokkos::View<Beams::ElemIndices*> elem_indices;  // Element indices
    View_NxN shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx7 node_pos_rot_;                          // Node global position/rotation vector
    View_Nx3 qp_pos_deriv_;                          // quadrature point position derivative
    View_N qp_jacobian_;                             // Jacobians

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_pos_deriv = Kokkos::subview(qp_pos_deriv_, idx.qp_range, Kokkos::ALL);
        auto node_pos = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        // Interpolate quadrature point position derivative from node position
        InterpVector3(shape_deriv, node_pos, qp_pos_deriv);

        //  Loop through quadrature points
        for (size_t j = 0; j < idx.num_qps; ++j) {
            // Calculate Jacobian as norm of derivative
            qp_jacobian(j) = Kokkos::sqrt(
                Kokkos::pow(qp_pos_deriv(j, 0), 2.) + Kokkos::pow(qp_pos_deriv(j, 1), 2.) +
                Kokkos::pow(qp_pos_deriv(j, 2), 2.)
            );

            // Apply Jacobian to row
            qp_pos_deriv(j, 0) /= qp_jacobian(j);
            qp_pos_deriv(j, 1) /= qp_jacobian(j);
            qp_pos_deriv(j, 2) /= qp_jacobian(j);
        }
    }
};

struct InterpolateQPState {
    Kokkos::View<Beams::ElemIndices*> elem_indices;
    View_NxN shape_interp_;
    View_NxN shape_deriv_;
    View_N qp_jacobian_;
    View_Nx7 node_u_;
    View_Nx3 qp_u_;
    View_Nx3 qp_u_prime_;
    View_Nx4 qp_r_;
    View_Nx4 qp_r_prime_;

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto idx = elem_indices(i_elem);
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_u = Kokkos::subview(node_u_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u = Kokkos::subview(qp_u_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_u, qp_u);

        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);
        auto qp_u_prime = Kokkos::subview(qp_u_prime_, idx.qp_range, Kokkos::ALL);
        InterpVector3Deriv(shape_deriv, qp_jacobian, node_u, qp_u_prime);

        auto node_r = Kokkos::subview(node_u_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_r = Kokkos::subview(qp_r_, idx.qp_range, Kokkos::ALL);
        InterpQuaternion(shape_interp, node_r, qp_r);

        auto qp_r_prime = Kokkos::subview(qp_r_prime_, idx.qp_range, Kokkos::ALL);
        InterpVector4Deriv(shape_deriv, qp_jacobian, node_r, qp_r_prime);
    }
};

struct InterpolateQPVelocity {
    Kokkos::View<Beams::ElemIndices*> elem_indices;  // Element indices
    View_NxN shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_N qp_jacobian_;                             // Num Nodes x Num Quadrature points
    View_Nx6 node_u_dot_;                            // Node translation & angular velocity
    View_Nx3 qp_u_dot_;                              // qp translation velocity
    View_Nx3 qp_omega_;                              // qp angular velocity

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_u_dot = Kokkos::subview(node_u_dot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u_dot = Kokkos::subview(qp_u_dot_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_u_dot, qp_u_dot);

        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);
        auto node_omega = Kokkos::subview(node_u_dot_, idx.node_range, Kokkos::make_pair(3, 6));
        auto qp_omega = Kokkos::subview(qp_omega_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_omega, qp_omega);
    }
};

struct InterpolateQPAcceleration {
    Kokkos::View<Beams::ElemIndices*> elem_indices;  // Element indices
    View_NxN shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_N qp_jacobian_;                             // Num Nodes x Num Quadrature points
    View_Nx6 node_u_ddot_;                           // Node translation & angular velocity
    View_Nx3 qp_u_ddot_;                             // qp translation velocity
    View_Nx3 qp_omega_dot_;                          // qp angular velocity

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_u_ddot = Kokkos::subview(node_u_ddot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_u_ddot = Kokkos::subview(qp_u_ddot_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_u_ddot, qp_u_ddot);

        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);
        auto node_omega_dot = Kokkos::subview(node_u_ddot_, idx.node_range, Kokkos::make_pair(3, 6));
        auto qp_omega_dot = Kokkos::subview(qp_omega_dot_, idx.qp_range, Kokkos::ALL);
        InterpVector3(shape_interp, node_omega_dot, qp_omega_dot);
    }
};

struct CalculateRR0 {
    View_Nx4 qp_r0_;     // quadrature point initial rotation
    View_Nx4 qp_r_;      // quadrature rotation displacement
    View_Nx6x6 qp_RR0_;  // quadrature global rotation

    KOKKOS_FUNCTION void operator()(const size_t i_qp) const {
        Quaternion R(qp_r_(i_qp, 0), qp_r_(i_qp, 1), qp_r_(i_qp, 2), qp_r_(i_qp, 3));
        Quaternion R0(qp_r0_(i_qp, 0), qp_r0_(i_qp, 1), qp_r0_(i_qp, 2), qp_r0_(i_qp, 3));
        auto RR0 = (R * R0).to_rotation_matrix();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                qp_RR0_(i_qp, i, j) = RR0(i, j);
                qp_RR0_(i_qp, 3 + i, 3 + j) = RR0(i, j);
            }
        }
    }
};

struct CalculateMuu {
    View_Nx6x6 qp_RR0_;    //
    View_Nx6x6 qp_Mstar_;  //
    View_Nx6x6 qp_Muu_;    //
    View_Nx6x6 qp_Mtmp_;   //

    KOKKOS_FUNCTION
    void operator()(const size_t i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mstar = Kokkos::subview(qp_Mstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mtmp = Kokkos::subview(qp_Mtmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Mstar, Mtmp);
        MatMulABT(Mtmp, RR0, Muu);
    }
};

struct CalculateCuu {
    View_Nx6x6 qp_RR0_;    //
    View_Nx6x6 qp_Cstar_;  //
    View_Nx6x6 qp_Cuu_;    //
    View_Nx6x6 qp_Ctmp_;   //

    KOKKOS_FUNCTION
    void operator()(const size_t i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cstar = Kokkos::subview(qp_Cstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Ctmp = Kokkos::subview(qp_Ctmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Cstar, Ctmp);
        MatMulABT(Ctmp, RR0, Cuu);
    }
};

struct CalculateStrain {
    View_Nx3 qp_x0_prime_;  //
    View_Nx3 qp_u_prime_;   //
    View_Nx4 qp_r_;         //
    View_Nx4 qp_r_prime_;   //
    View_Nx6 qp_strain_;    //

    KOKKOS_FUNCTION
    void operator()(const size_t i_qp) const {
        Vector x0_prime(qp_x0_prime_(i_qp, 0), qp_x0_prime_(i_qp, 1), qp_x0_prime_(i_qp, 2));
        Vector u_prime(qp_u_prime_(i_qp, 0), qp_u_prime_(i_qp, 1), qp_u_prime_(i_qp, 2));
        Quaternion R(qp_r_(i_qp, 0), qp_r_(i_qp, 1), qp_r_(i_qp, 2), qp_r_(i_qp, 3));

        auto R_x0_prime = R * x0_prime;
        auto e1 = x0_prime + u_prime - R_x0_prime;

        double E[3][4];
        QuaternionDerivative(R, E);
        double R_prime[4] = {
            qp_r_prime_(i_qp, 0), qp_r_prime_(i_qp, 1), qp_r_prime_(i_qp, 2), qp_r_prime_(i_qp, 3)};
        double e2[3];

        for (size_t i = 0; i < 3; ++i) {
            e2[i] = 0.;
            for (size_t k = 0; k < 4; ++k) {
                e2[i] += E[i][k] * R_prime[k];
            }
        }

        qp_strain_(i_qp, 0) = e1.GetX();
        qp_strain_(i_qp, 1) = e1.GetY();
        qp_strain_(i_qp, 2) = e1.GetZ();
        qp_strain_(i_qp, 3) = 2.0 * e2[0];
        qp_strain_(i_qp, 4) = 2.0 * e2[1];
        qp_strain_(i_qp, 5) = 2.0 * e2[2];
    }
};

struct CalculateMassMatrixComponents {
    View_Nx6x6 qp_Muu_;
    View_Nx3 eta_;
    View_Nx3x3 rho_;
    View_Nx3x3 eta_tilde_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);
        if (m == 0.) {
            eta(0) = 0.;
            eta(1) = 0.;
            eta(2) = 0.;
        } else {
            eta(0) = Muu(5, 1) / m;
            eta(1) = -Muu(5, 0) / m;
            eta(2) = Muu(4, 0) / m;
        }
        for (size_t i = 0; i < rho.extent(0); ++i) {
            for (size_t j = 0; j < rho.extent(1); ++j) {
                rho(i, j) = Muu(i + 3, j + 3);
            }
        }
        VecTilde(eta, eta_tilde);
    }
};

struct CalculateTemporaryVariables {
    View_Nx3 qp_x0_prime_;
    View_Nx3 qp_u_prime_;
    View_Nx3x3 x0pupSS_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        Vector x0_prime(qp_x0_prime_(i_qp, 0), qp_x0_prime_(i_qp, 1), qp_x0_prime_(i_qp, 2));
        Vector u_prime(qp_u_prime_(i_qp, 0), qp_u_prime_(i_qp, 1), qp_u_prime_(i_qp, 2));
        auto x0pup = x0_prime + u_prime;
        double tmp[3][3];
        x0pup.Tilde(tmp);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                x0pupSS_(i_qp, i, j) = tmp[i][j];
            }
        }
    }
};

struct CalculateForceFC {
    View_Nx6x6 qp_Cuu_;
    View_Nx6 qp_strain_;
    View_Nx6 qp_FC_;
    View_Nx3x3 M_tilde_;
    View_Nx3x3 N_tilde_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto strain = Kokkos::subview(qp_strain_, i_qp, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_qp, Kokkos::ALL);
        auto M_tilde = Kokkos::subview(M_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);

        MatVecMulAB(Cuu, strain, FC);
        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        auto M = Kokkos::subview(FC, Kokkos::make_pair(3, 6));
        VecTilde(M, M_tilde);
        VecTilde(N, N_tilde);
    }
};

struct CalculateForceFD {
    View_Nx3x3 x0pupSS_;
    View_Nx6 qp_FC_;
    View_Nx6 qp_FD_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_qp, Kokkos::ALL);
        auto FD = Kokkos::subview(qp_FD_, i_qp, Kokkos::ALL);

        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        for (size_t i = 0; i < FD.extent(0); ++i) {
            FD(i) = 0.;
        }
        MatVecMulATB(x0pupSS, N, Kokkos::subview(FD, Kokkos::make_pair(3, 6)));
    }
};

struct CalculateInertialForces {
    View_Nx6x6 qp_Muu_;
    View_Nx3 qp_u_ddot_;
    View_Nx3 qp_omega_;
    View_Nx3 qp_omega_dot_;
    View_Nx3x3 eta_tilde_;
    View_Nx3x3 omega_tilde_;
    View_Nx3x3 omega_dot_tilde_;
    View_Nx3x3 rho_;
    View_Nx3 eta_;
    View_Nx3 v1_;
    View_Nx3x3 M1_;
    View_Nx6 qp_FI_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto u_ddot = Kokkos::subview(qp_u_ddot_, i_qp, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_dot = Kokkos::subview(qp_omega_dot_, i_qp, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_dot_tilde = Kokkos::subview(omega_dot_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FI = Kokkos::subview(qp_FI_, i_qp, Kokkos::ALL);

        auto m = Muu(0, 0);
        VecTilde(omega, omega_tilde);
        VecTilde(omega_dot, omega_dot_tilde);
        auto FI_1 = Kokkos::subview(FI, Kokkos::make_pair(0, 3));
        MatMulAB(omega_tilde, omega_tilde, M1);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                M1(i, j) += omega_dot_tilde(i, j);
                M1(i, j) *= m;
            }
        }
        MatVecMulAB(M1, eta, FI_1);
        for (size_t i = 0; i < 3; i++) {
            FI_1(i) += u_ddot(i) * m;
        }
        auto FI_2 = Kokkos::subview(FI, Kokkos::make_pair(3, 6));
        VecScale(u_ddot, m, V1);
        MatVecMulAB(eta_tilde, V1, FI_2);
        MatVecMulAB(rho, omega_dot, V1);
        for (size_t i = 0; i < 3; i++) {
            FI_2(i) += V1(i);
        }
        MatMulAB(omega_tilde, rho, M1);
        MatVecMulAB(M1, omega, V1);
        for (size_t i = 0; i < 3; i++) {
            FI_2(i) += V1(i);
        }
    }
};

struct CalculateGravityForce {
    View_3 gravity;
    View_Nx6x6 qp_Muu_;
    View_Nx3x3 eta_tilde_;
    View_Nx3 v1_;
    View_Nx6 qp_FG_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto FG = Kokkos::subview(qp_FG_, i_qp, Kokkos::ALL);
        auto m = Muu(0, 0);
        VecScale(gravity, m, V1);
        for (size_t i = 0; i < 3; ++i) {
            FG(i) = V1(i);
        }
        MatVecMulAB(eta_tilde, V1, Kokkos::subview(FG, Kokkos::make_pair(3, 6)));
    }
};

struct CalculateOuu {
    View_Nx6x6 qp_Cuu_;
    View_Nx3x3 x0pupSS_;
    View_Nx3x3 M_tilde_;
    View_Nx3x3 N_tilde_;
    View_Nx6x6 qp_Ouu_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M_tilde = Kokkos::subview(M_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Ouu = Kokkos::subview(qp_Ouu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto C21 = Kokkos::subview(Cuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
        for (size_t i = 0; i < Ouu.extent(0); ++i) {
            for (size_t j = 0; j < Ouu.extent(1); ++j) {
                Ouu(i, j) = 0.;
            }
        }
        auto Ouu_12 = Kokkos::subview(Ouu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        auto Ouu_22 = Kokkos::subview(Ouu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, Ouu_12);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Ouu_12(i, j) -= N_tilde(i, j);
            }
        }
        MatMulAB(C21, x0pupSS, Ouu_22);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Ouu_22(i, j) -= M_tilde(i, j);
            }
        }
    }
};

struct CalculatePuu {
    View_Nx6x6 qp_Cuu_;
    View_Nx3x3 x0pupSS_;
    View_Nx3x3 N_tilde_;
    View_Nx6x6 qp_Puu_;
    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Puu = Kokkos::subview(qp_Puu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto C12 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        for (size_t i = 0; i < Puu.extent(0); ++i) {
            for (size_t j = 0; j < Puu.extent(1); ++j) {
                Puu(i, j) = 0.;
            }
        }
        auto Puu_21 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
        MatMulATB(x0pupSS, C11, Puu_21);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Puu_21(i, j) += N_tilde(i, j);
            }
        }
        auto Puu_22 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulATB(x0pupSS, C12, Puu_22);
    }
};

struct CalculateQuu {
    View_Nx6x6 qp_Cuu_;
    View_Nx3x3 x0pupSS_;
    View_Nx3x3 N_tilde_;
    View_Nx3x3 M1_;
    View_Nx6x6 qp_Quu_;
    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Quu = Kokkos::subview(qp_Quu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        for (size_t i = 0; i < Quu.extent(0); ++i) {
            for (size_t j = 0; j < Quu.extent(1); ++j) {
                Quu(i, j) = 0.;
            }
        }
        auto Quu_22 = Kokkos::subview(Quu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, M1);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                M1(i, j) -= N_tilde(i, j);
            }
        }
        MatMulATB(x0pupSS, M1, Quu_22);
    }
};

struct CalculateGyroscopicMatrix {
    View_Nx6x6 qp_Muu_;
    View_Nx3 qp_omega_;
    View_Nx3x3 omega_tilde_;
    View_Nx3x3 rho_;
    View_Nx3 eta_;
    View_Nx3 v1_;
    View_Nx3 v2_;
    View_Nx3x3 M1_;
    View_Nx6x6 qp_Guu_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto V2 = Kokkos::subview(v2_, i_qp, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Guu = Kokkos::subview(qp_Guu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);
        // Inertia gyroscopic matrix
        for (size_t i = 0; i < Guu.extent(0); ++i) {
            for (size_t j = 0; j < Guu.extent(1); ++j) {
                Guu(i, j) = 0.;
            }
        }
        // omega.tilde() * m * eta.tilde().t() + (omega.tilde() * m * eta).tilde().t()
        auto Guu_12 = Kokkos::subview(Guu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        VecScale(eta, m, V1);
        VecTilde(V1, M1);
        MatMulABT(omega_tilde, M1, Guu_12);
        MatVecMulAB(omega_tilde, V1, V2);
        VecTilde(V2, M1);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Guu_12(i, j) += M1(j, i);
            }
        }
        // Guu_22 = omega.tilde() * rho - (rho * omega).tilde()
        auto Guu_22 = Kokkos::subview(Guu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(omega_tilde, rho, Guu_22);
        MatVecMulAB(rho, omega, V1);
        VecTilde(V1, M1);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Guu_22(i, j) -= M1(i, j);
            }
        }
    }
};

struct CalculateInertiaStiffnessMatrix {
    View_Nx6x6 qp_Muu_;
    View_Nx3 qp_u_ddot_;
    View_Nx3 qp_omega_;
    View_Nx3 qp_omega_dot_;
    View_Nx3x3 omega_tilde_;
    View_Nx3x3 omega_dot_tilde_;
    View_Nx3x3 rho_;
    View_Nx3 eta_;
    View_Nx3 v1_;
    View_Nx3x3 M1_;
    View_Nx3x3 M2_;
    View_Nx6x6 qp_Kuu_;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto u_ddot = Kokkos::subview(qp_u_ddot_, i_qp, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_dot = Kokkos::subview(qp_omega_dot_, i_qp, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_dot_tilde = Kokkos::subview(omega_dot_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M2 = Kokkos::subview(M2_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Kuu = Kokkos::subview(qp_Kuu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);

        // Inertia stiffness matrix
        for (size_t i = 0; i < Kuu.extent(0); ++i) {
            for (size_t j = 0; j < Kuu.extent(1); ++j) {
                Kuu(i, j) = 0.;
            }
        }
        auto Kuu_12 = Kokkos::subview(Kuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        MatMulAB(omega_tilde, omega_tilde, M1);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                M1(i, j) += omega_dot_tilde(i, j);
            }
        }
        VecScale(eta, m, V1);
        VecTilde(V1, M2);
        MatMulABT(M1, M2, Kuu_12);
        auto Kuu_22 = Kokkos::subview(Kuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        VecTilde(u_ddot, M1);
        VecScale(eta, m, V1);
        VecTilde(V1, M2);
        MatMulAB(M1, M2, Kuu_22);
        MatMulAB(rho, omega_dot_tilde, M1);
        MatVecMulAB(rho, omega_dot, V1);
        VecTilde(V1, M2);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Kuu_22(i, j) += M1(i, j) - M2(i, j);
            }
        }
        MatMulAB(rho, omega_tilde, M1);
        MatVecMulAB(rho, omega, V1);
        VecTilde(V1, M2);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                M1(i, j) -= M2(i, j);
            }
        }
        MatMulAB(omega_tilde, M1, M2);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                Kuu_22(i, j) += M2(i, j);
            }
        }
    }
};

struct CalculateNodeForces {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fc_;                                 //
    View_Nx6::const_type qp_Fd_;                                 //
    View_Nx6::const_type qp_Fi_;                                 //
    View_Nx6::const_type qp_Fg_;                                 //
    View_Nx6 node_FE_;                                           // Elastic force
    View_Nx6 node_FI_;                                           // Inertial force
    View_Nx6 node_FG_;                                           // Gravity force

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        auto idx = elem_indices[i_elem];

        for (size_t i = idx.node_range.first; i < idx.node_range.second; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                node_FE_(i, j) = 0.;
                node_FG_(i, j) = 0.;
                node_FI_(i, j) = 0.;
            }
        }

        for (size_t i_index = 0; i_index < idx.num_nodes; ++i_index) {    // Nodes
            for (size_t j_index = 0; j_index < idx.num_qps; ++j_index) {  // QPs
                const auto i = idx.node_range.first + i_index;
                const auto j = idx.qp_range.first + j_index;
                const auto weight = qp_weight_(j);
                const auto coeff_c = weight * shape_deriv_(i, j_index);
                const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
                const auto coeff_i = coeff_d;
                const auto coeff_g = coeff_d;
                for (size_t k = 0; k < 6; ++k) {  // Components
                    node_FE_(i, k) += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                }
                for (size_t k = 0; k < 6; ++k) {  // Components
                    node_FI_(i, k) += coeff_i * qp_Fi_(j, k);
                }
                for (size_t k = 0; k < 6; ++k) {  // Components
                    node_FG_(i, k) += coeff_g * qp_Fg_(j, k);
                }
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto idx = elem_indices(member.league_rank());
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(member, idx.num_nodes, 6),
            [=](size_t i_index, size_t j) {
                const auto i = idx.node_range.first + i_index;
                node_FE_(i, j) = 0.;
                node_FG_(i, j) = 0.;
                node_FI_(i, j) = 0.;
            }
        );

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, idx.num_nodes), [=](size_t i_index) {
            for (size_t j_index = 0; j_index < idx.num_qps; ++j_index) {  // QPs
                const auto i = idx.node_range.first + i_index;
                const auto j = idx.qp_range.first + j_index;
                const auto weight = qp_weight_(j);
                const auto coeff_c = weight * shape_deriv_(i, j_index);
                const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
                const auto coeff_i = coeff_d;
                const auto coeff_g = coeff_d;
                for (size_t k = 0; k < 6; ++k) {  // Components
                    node_FE_(i, k) += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                }
                for (size_t k = 0; k < 6; ++k) {  // Components
                    node_FI_(i, k) += coeff_i * qp_Fi_(j, k);
                }
                for (size_t k = 0; k < 6; ++k) {  // Components
                    node_FG_(i, k) += coeff_g * qp_Fg_(j, k);
                }
            }
        });
    }
};

struct IntegrateMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<size_t*>::const_type node_state_indices;        // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6x6::const_type qp_M_;                                //
    View_NxN_atomic gbl_M_;                                      //

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        const auto idx = elem_indices[i_elem];

        for (size_t i = idx.node_range.first; i < idx.node_range.second; ++i) {      // Nodes
            for (size_t j = idx.node_range.first; j < idx.node_range.second; ++j) {  // Nodes
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (size_t k = 0; k < idx.num_qps; ++k) {
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto coeff = w * phi_i * phi_j * jacobian;
                    for (size_t m = 0; m < kLieAlgebraComponents; ++m) {
                        for (size_t n = 0; n < kLieAlgebraComponents; ++n) {
                            local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                        }
                    }
                }
                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                for (size_t m = 0; m < kLieAlgebraComponents; ++m) {
                    for (size_t n = 0; n < kLieAlgebraComponents; ++n) {
                        gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                    }
                }
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto idx = elem_indices(member.league_rank());
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes),
            [=](size_t i_index, size_t j_index) {
                const auto i = i_index + idx.node_range.first;
                const auto j = j_index + idx.node_range.first;
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (size_t k = 0; k < idx.num_qps; ++k) {
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto coeff = w * phi_i * phi_j * jacobian;
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorMDRange(member, 6, 6),
                        [=](size_t m, size_t n) {
                            local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                        }
                    );
                }
                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorMDRange(member, 6, 6),
                    [=](size_t m, size_t n) {
                        gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                    }
                );
            }
        );
    }
};

struct IntegrateElasticStiffnessMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<size_t*>::const_type node_state_indices;        // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6x6::const_type qp_Puu_;                              //
    View_Nx6x6::const_type qp_Cuu_;                              //
    View_Nx6x6::const_type qp_Ouu_;                              //
    View_Nx6x6::const_type qp_Quu_;                              //
    View_NxN_atomic gbl_M_;

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        const auto idx = elem_indices[i_elem];

        for (size_t i = idx.node_range.first; i < idx.node_range.second; ++i) {      // Nodes
            for (size_t j = idx.node_range.first; j < idx.node_range.second; ++j) {  // Nodes
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (size_t k = 0; k < idx.num_qps; ++k) {  // QPs
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto phi_prime_i = shape_deriv_(i, k);
                    const auto phi_prime_j = shape_deriv_(j, k);
                    const auto coeff_P = w * (phi_i * phi_prime_j);
                    const auto coeff_Q = w * (phi_i * phi_j * jacobian);
                    const auto coeff_C = w * (phi_prime_i * phi_prime_j / jacobian);
                    const auto coeff_O = w * (phi_prime_i * phi_j);
                    for (size_t m = 0; m < 6; ++m) {      // Matrix components
                        for (size_t n = 0; n < 6; ++n) {  // Matrix components
                            local_M(m, n) +=
                                coeff_P * qp_Puu_(k_qp, m, n) + coeff_Q * qp_Quu_(k_qp, m, n) +
                                coeff_C * qp_Cuu_(k_qp, m, n) + coeff_O * qp_Ouu_(k_qp, m, n);
                        }
                    }
                }

                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                for (size_t m = 0; m < 6; ++m) {
                    for (size_t n = 0; n < 6; ++n) {
                        gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                    }
                }
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto idx = elem_indices(member.league_rank());
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes),
            [=](size_t i_index, size_t j_index) {
                const auto i = i_index + idx.node_range.first;
                const auto j = j_index + idx.node_range.first;
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (size_t k = 0; k < idx.num_qps; ++k) {
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto phi_prime_i = shape_deriv_(i, k);
                    const auto phi_prime_j = shape_deriv_(j, k);
                    const auto coeff_P = w * (phi_i * phi_prime_j);
                    const auto coeff_Q = w * (phi_i * phi_j * jacobian);
                    const auto coeff_C = w * (phi_prime_i * phi_prime_j / jacobian);
                    const auto coeff_O = w * (phi_prime_i * phi_j);
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorMDRange(member, 6, 6),
                        [=](size_t m, size_t n) {
                            local_M(m, n) +=
                                coeff_P * qp_Puu_(k_qp, m, n) + coeff_Q * qp_Quu_(k_qp, m, n) +
                                coeff_C * qp_Cuu_(k_qp, m, n) + coeff_O * qp_Ouu_(k_qp, m, n);
                        }
                    );
                }

                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorMDRange(member, 6, 6),
                    [=](size_t m, size_t n) {
                        gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                    }
                );
            }
        );
    }
};

struct IntegrateResidualVector {
    Kokkos::View<size_t*> node_state_indices_;
    View_Nx6::const_type node_FE_;  // Elastic force
    View_Nx6::const_type node_FI_;  // Inertial force
    View_Nx6::const_type node_FG_;  // Gravity force
    View_Nx6::const_type node_FX_;  // External force
    View_N_atomic residual_vector_;

    IntegrateResidualVector(
        Kokkos::View<size_t*> node_state_indices, View_Nx6 node_FE, View_Nx6 node_FI,
        View_Nx6 node_FG, View_Nx6 node_FX, View_N residual_vector
    )
        : node_state_indices_(node_state_indices),
          node_FE_(node_FE),
          node_FI_(node_FI),
          node_FG_(node_FG),
          node_FX_(node_FX),
          residual_vector_(residual_vector) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i_node) const {
        auto i_rv_start = node_state_indices_(i_node) * kLieAlgebraComponents;
        for (size_t j = 0; j < 6; j++) {
            residual_vector_(i_rv_start + j) += node_FE_(i_node, j) + node_FI_(i_node, j) -
                                                node_FX_(i_node, j) - node_FG_(i_node, j);
        }
    }
};

}  // namespace openturbine
