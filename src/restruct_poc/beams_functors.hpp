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
    for (int i = 0; i < A.extent_int(0); ++i) {
        A(i) = value;
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulAB(View_A A, View_B B, View_C C) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        auto local_result = 0.;
        for (int k = 0; k < B.extent_int(0); ++k) {
            local_result += A(i, k) * B(k);
        }
        C(i) = local_result;
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatVecMulATB(View_A A, View_B B, View_C C) {
    for (int i = 0; i < A.extent_int(1); ++i) {
        auto local_result = 0.;
        for (int k = 0; k < B.extent_int(0); ++k) {
            local_result += A(k, i) * B(k);
        }
        C(i) = local_result;
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
    for (int i = 0; i < v_in.extent_int(0); ++i) {
        v_out(i) = v_in(i) * scale;
    }
}

//------------------------------------------------------------------------------
// Matrix functions
//------------------------------------------------------------------------------

template <typename View_A>
KOKKOS_INLINE_FUNCTION void FillMatrix(View_A A, double value) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        for (int j = 0; j < A.extent_int(1); ++j) {
            A(i, j) = value;
        }
    }
}

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION void MatScale(A m_in, double scale, B m_out) {
    for (int i = 0; i < m_in.extent_int(0); ++i) {
        for (int j = 0; j < m_in.extent_int(1); ++j) {
            m_out(i, j) = m_in(i, j) * scale;
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatAdd(View_A M_A, View_B M_B, View_C M_C) {
    for (int i = 0; i < M_A.extent_int(0); ++i) {
        for (int j = 0; j < M_A.extent_int(1); ++j) {
            M_C(i, j) = M_A(i, j) + M_B(i, j);
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulAB(View_A A, View_B B, View_C C) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        for (int j = 0; j < B.extent_int(1); ++j) {
            auto local_result = 0.;
            for (int k = 0; k < B.extent_int(0); ++k) {
                local_result += A(i, k) * B(k, j);
            }
            C(i, j) = local_result;
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulATB(View_A AT, View_B B, View_C C) {
    for (int i = 0; i < AT.extent_int(1); ++i) {
        for (int j = 0; j < B.extent_int(1); ++j) {
            auto local_result = 0.;
            for (int k = 0; k < B.extent_int(0); ++k) {
                local_result += AT(k, i) * B(k, j);
            }
            C(i, j) = local_result;
        }
    }
}

template <typename View_A, typename View_B, typename View_C>
KOKKOS_INLINE_FUNCTION void MatMulABT(View_A A, View_B BT, View_C C) {
    for (int i = 0; i < A.extent_int(0); ++i) {
        for (int j = 0; j < BT.extent_int(0); ++j) {
            auto local_result = 0.;
            for (int k = 0; k < BT.extent_int(1); ++k) {
                local_result += A(i, k) * BT(j, k);
            }
            C(i, j) = local_result;
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
void InterpVector3(View_NxN::const_type shape_matrix, View_Nx3::const_type node_v, View_Nx3 qp_v) {
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i = 0; i < node_v.extent_int(0); ++i) {
            const auto phi = shape_matrix(i, j);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_v(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_v(j, k) = local_total[k];
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector4(View_NxN::const_type shape_matrix, View_Nx4::const_type node_v, View_Nx4 qp_v) {
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i = 0; i < node_v.extent_int(0); ++i) {
            const auto phi = shape_matrix(i, j);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_v(i, k) * phi;
            }
        }
        for (int k = 0; k < 4; ++k) {
            qp_v(j, k) = local_total[k];
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpQuaternion(
    View_NxN::const_type shape_matrix, View_Nx4::const_type node_v, View_Nx4 qp_v
) {
    InterpVector4(shape_matrix, node_v, qp_v);
    static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
    // Normalize quaternions (rows)
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        auto length = Kokkos::sqrt(
            Kokkos::pow(qp_v(j, 0), 2) + Kokkos::pow(qp_v(j, 1), 2) + Kokkos::pow(qp_v(j, 2), 2) +
            Kokkos::pow(qp_v(j, 3), 2)
        );
        if (length == 0.) {
            for (int k = 0; k < 4; ++k) {
                qp_v(j, k) = length_zero_result[k];
            }
        } else {
            for (int k = 0; k < 4; ++k) {
                qp_v(j, k) /= length;
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector3Deriv(
    View_NxN::const_type shape_matrix_deriv, View_N::const_type jacobian,
    View_Nx3::const_type node_v, View_Nx3 qp_v
) {
    InterpVector3(shape_matrix_deriv, node_v, qp_v);
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        const auto jac = jacobian(j);
        for (int k = 0; k < qp_v.extent_int(1); ++k) {
            qp_v(j, k) /= jac;
        }
    }
}

KOKKOS_INLINE_FUNCTION
void InterpVector4Deriv(
    View_NxN::const_type shape_matrix_deriv, View_N::const_type jacobian,
    View_Nx4::const_type node_v, View_Nx4 qp_v
) {
    InterpVector4(shape_matrix_deriv, node_v, qp_v);
    for (int j = 0; j < qp_v.extent_int(0); ++j) {
        const auto jac = jacobian(j);
        for (int k = 0; k < qp_v.extent_int(1); ++k) {
            qp_v(j, k) /= jac;
        }
    }
}

//------------------------------------------------------------------------------
// Functors to perform calculations on Beams structure
//------------------------------------------------------------------------------

struct InterpolateQPPosition {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_pos_rot_;                          // Node global position vector
    View_Nx3 qp_pos_;                                            // quadrature point position

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_pos = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_pos = Kokkos::subview(qp_pos_, idx.qp_range, Kokkos::ALL);

        // Perform matrix-matrix multiplication
        for (int j = 0; j < idx.num_qps; ++j) {
            auto local_result = Kokkos::Array<double, 3>{};
            for (int i = 0; i < idx.num_nodes; ++i) {
                const auto phi = shape_interp(i, j);
                for (int k = 0; k < kVectorComponents; ++k) {
                    local_result[k] += node_pos(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_pos(j, k) = local_result[k];
            }
        }
    }
};

struct InterpolateQPRotation {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_pos_rot_;                          // Node global position vector
    View_Nx4 qp_rot_;                                            // quadrature point rotation

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_rot = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_rot = Kokkos::subview(qp_rot_, idx.qp_range, Kokkos::ALL);

        InterpQuaternion(shape_interp, node_rot, qp_rot);
    }
};

struct InterpolateQPRotationDerivative {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_Nx7::const_type node_pos_rot_;  // Node global position/rotation vector
    View_Nx4 qp_rot_deriv_;              // quadrature point rotation derivative

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_rot_deriv = Kokkos::subview(qp_rot_deriv_, idx.qp_range, Kokkos::ALL);
        auto node_rot = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        InterpVector4Deriv(shape_deriv, qp_jacobian, node_rot, qp_rot_deriv);
    }
};

struct CalculateJacobian {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_pos_rot_;  // Node global position/rotation vector
    View_Nx3 qp_pos_deriv_;              // quadrature point position derivative
    View_N qp_jacobian_;                 // Jacobians

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_pos_deriv = Kokkos::subview(qp_pos_deriv_, idx.qp_range, Kokkos::ALL);
        auto node_pos = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        // Interpolate quadrature point position derivative from node position
        InterpVector3(shape_deriv, node_pos, qp_pos_deriv);

        //  Loop through quadrature points
        for (int j = 0; j < idx.num_qps; ++j) {
            // Calculate Jacobian as norm of derivative
            const auto jacobian = Kokkos::sqrt(
                Kokkos::pow(qp_pos_deriv(j, 0), 2.) + Kokkos::pow(qp_pos_deriv(j, 1), 2.) +
                Kokkos::pow(qp_pos_deriv(j, 2), 2.)
            );
            qp_jacobian(j) = jacobian;
            // Apply Jacobian to row
            for (int k = 0; k < 3; ++k) {
                qp_pos_deriv(j, k) /= jacobian;
            }
        }
    }
};

struct InterpolateQPU {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_interp_;
    View_Nx7::const_type node_u_;
    View_Nx3 qp_u_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }
        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPU_Prime {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_deriv_;
    View_N::const_type qp_jacobian_;
    View_Nx7::const_type node_u_;
    View_Nx3 qp_u_prime_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_prime_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        const auto jacobian = qp_jacobian_(j);
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto dphi = shape_deriv_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_(i, k) * dphi / jacobian;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_prime_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPR {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_interp_;
    View_Nx7::const_type node_u_;
    View_Nx4 qp_r_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * phi;
                }
            }
            const auto length = Kokkos::sqrt(
                local_total[0] * local_total[0] + local_total[1] * local_total[1] +
                local_total[2] * local_total[2] + local_total[3] * local_total[3]
            );
            static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
            if (length == 0.) {
                local_total = length_zero_result;
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_u_(i, k + 3) * phi;
            }
        }
        const auto length = Kokkos::sqrt(
            local_total[0] * local_total[0] + local_total[1] * local_total[1] +
            local_total[2] * local_total[2] + local_total[3] * local_total[3]
        );
        static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        if (length == 0.) {
            local_total = length_zero_result;
        }
        for (int k = 0; k < 4; ++k) {
            qp_r_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPR_Prime {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_deriv_;
    View_N::const_type qp_jacobian_;
    View_Nx7::const_type node_u_;
    View_Nx4 qp_r_prime_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_prime_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        const auto jacobian = qp_jacobian_(j);
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto dphi = shape_deriv_(i, j_index);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_u_(i, k + 3) * dphi / jacobian;
            }
        }
        for (int k = 0; k < 4; ++k) {
            qp_r_prime_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPState {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type shape_interp_;
    View_NxN::const_type shape_deriv_;
    View_N::const_type qp_jacobian_;
    View_Nx7::const_type node_u_;
    View_Nx3 qp_u_;
    View_Nx3 qp_u_prime_;
    View_Nx4 qp_r_;
    View_Nx4 qp_r_prime_;

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_(j, k) = local_total[k];
            }
        }

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_(i, k) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_prime_(j, k) = local_total[k];
            }
        }

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * phi;
                }
            }
            const auto length = Kokkos::sqrt(
                local_total[0] * local_total[0] + local_total[1] * local_total[1] +
                local_total[2] * local_total[2] + local_total[3] * local_total[3]
            );
            static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
            if (length == 0.) {
                local_total = length_zero_result;
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_(j, k) = local_total[k];
            }
        }

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto jacobian = qp_jacobian_(j);
            auto local_total = Kokkos::Array<double, 4>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto dphi = shape_deriv_(i, j_index);
                for (int k = 0; k < 4; ++k) {
                    local_total[k] += node_u_(i, k + 3) * dphi / jacobian;
                }
            }
            for (int k = 0; k < 4; ++k) {
                qp_r_prime_(j, k) = local_total[k];
            }
        }
    }
};

struct InterpolateQPVelocity_Translation {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6 node_u_dot_;  // Node translation & angular velocity
    View_Nx3 qp_u_dot_;    // qp translation velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_dot_(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_dot_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_dot_(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_dot_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPVelocity_Angular {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                           // Num Nodes x Num Quadrature points
    View_Nx6 node_u_dot_;  // Node translation & angular velocity
    View_Nx3 qp_omega_;    // qp angular velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_dot_(i, k + 3) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_omega_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_dot_(i, k + 3) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_omega_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPVelocity {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_N::const_type qp_jacobian_;                             // Num Nodes x Num Quadrature points
    View_Nx6 node_u_dot_;  // Node translation & angular velocity
    View_Nx3 qp_u_dot_;    // qp translation velocity
    View_Nx3 qp_omega_;    // qp angular velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
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

struct InterpolateQPAcceleration_Translation {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6::const_type node_u_ddot_;  // Node translation & angular velocity
    View_Nx3 qp_u_ddot_;                // qp translation velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_ddot_(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_ddot_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_ddot_(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_ddot_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPAcceleration_Angular {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6::const_type node_u_ddot_;  // Node translation & angular velocity
    View_Nx3 qp_omega_dot_;             // qp angular velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_ddot_(i, k + 3) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_omega_dot_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_ddot_(i, k + 3) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_omega_dot_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPAcceleration {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_N::const_type qp_jacobian_;                             // Num Nodes x Num Quadrature points
    View_Nx6::const_type node_u_ddot_;  // Node translation & angular velocity
    View_Nx3 qp_u_ddot_;                // qp translation velocity
    View_Nx3 qp_omega_dot_;             // qp angular velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
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
    View_Nx4::const_type qp_r0_;  // quadrature point initial rotation
    View_Nx4::const_type qp_r_;   // quadrature rotation displacement
    View_Nx6x6 qp_RR0_;           // quadrature global rotation

    KOKKOS_FUNCTION void operator()(const int i_qp) const {
        Quaternion R(qp_r_(i_qp, 0), qp_r_(i_qp, 1), qp_r_(i_qp, 2), qp_r_(i_qp, 3));
        Quaternion R0(qp_r0_(i_qp, 0), qp_r0_(i_qp, 1), qp_r0_(i_qp, 2), qp_r0_(i_qp, 3));
        auto RR0 = (R * R0).to_rotation_matrix();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                qp_RR0_(i_qp, i, j) = RR0(i, j);
                qp_RR0_(i_qp, 3 + i, 3 + j) = RR0(i, j);
            }
        }
    }
};

struct CalculateMuu {
    View_Nx6x6::const_type qp_RR0_;    //
    View_Nx6x6::const_type qp_Mstar_;  //
    View_Nx6x6 qp_Muu_;                //
    View_Nx6x6 qp_Mtmp_;               //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mstar = Kokkos::subview(qp_Mstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mtmp = Kokkos::subview(qp_Mtmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Mstar, Mtmp);
        MatMulABT(Mtmp, RR0, Muu);
    }
};

struct CalculateCuu {
    View_Nx6x6::const_type qp_RR0_;    //
    View_Nx6x6::const_type qp_Cstar_;  //
    View_Nx6x6 qp_Cuu_;                //
    View_Nx6x6 qp_Ctmp_;               //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cstar = Kokkos::subview(qp_Cstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Ctmp = Kokkos::subview(qp_Ctmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Cstar, Ctmp);
        MatMulABT(Ctmp, RR0, Cuu);
    }
};

struct CalculateStrain {
    View_Nx3::const_type qp_x0_prime_;  //
    View_Nx3::const_type qp_u_prime_;   //
    View_Nx4::const_type qp_r_;         //
    View_Nx4::const_type qp_r_prime_;   //
    View_Nx6 qp_strain_;                //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
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

        for (int i = 0; i < 3; ++i) {
            e2[i] = 0.;
            for (int k = 0; k < 4; ++k) {
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
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3 eta_;
    View_Nx3x3 rho_;
    View_Nx3x3 eta_tilde_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
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
        for (int i = 0; i < rho.extent_int(0); ++i) {
            for (int j = 0; j < rho.extent_int(1); ++j) {
                rho(i, j) = Muu(i + 3, j + 3);
            }
        }
        VecTilde(eta, eta_tilde);
    }
};

struct CalculateTemporaryVariables {
    View_Nx3::const_type qp_x0_prime_;
    View_Nx3::const_type qp_u_prime_;
    View_Nx3x3 x0pupSS_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        Vector x0_prime(qp_x0_prime_(i_qp, 0), qp_x0_prime_(i_qp, 1), qp_x0_prime_(i_qp, 2));
        Vector u_prime(qp_u_prime_(i_qp, 0), qp_u_prime_(i_qp, 1), qp_u_prime_(i_qp, 2));
        auto x0pup = x0_prime + u_prime;
        double tmp[3][3];
        x0pup.Tilde(tmp);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                x0pupSS_(i_qp, i, j) = tmp[i][j];
            }
        }
    }
};

struct CalculateForceFC {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx6::const_type qp_strain_;
    View_Nx6 qp_FC_;
    View_Nx3x3 M_tilde_;
    View_Nx3x3 N_tilde_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
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
    View_Nx3x3::const_type x0pupSS_;
    View_Nx6::const_type qp_FC_;
    View_Nx6 qp_FD_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_qp, Kokkos::ALL);
        auto FD = Kokkos::subview(qp_FD_, i_qp, Kokkos::ALL);

        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        for (int i = 0; i < FD.extent_int(0); ++i) {
            FD(i) = 0.;
        }
        MatVecMulATB(x0pupSS, N, Kokkos::subview(FD, Kokkos::make_pair(3, 6)));
    }
};

struct CalculateInertialForces {
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3::const_type qp_u_ddot_;
    View_Nx3::const_type qp_omega_;
    View_Nx3::const_type qp_omega_dot_;
    View_Nx3x3::const_type eta_tilde_;
    View_Nx3x3 omega_tilde_;
    View_Nx3x3 omega_dot_tilde_;
    View_Nx3x3::const_type rho_;
    View_Nx3::const_type eta_;
    View_Nx3 v1_;
    View_Nx3x3 M1_;
    View_Nx6 qp_FI_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
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
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M1(i, j) += omega_dot_tilde(i, j);
                M1(i, j) *= m;
            }
        }
        MatVecMulAB(M1, eta, FI_1);
        for (int i = 0; i < 3; i++) {
            FI_1(i) += u_ddot(i) * m;
        }
        auto FI_2 = Kokkos::subview(FI, Kokkos::make_pair(3, 6));
        VecScale(u_ddot, m, V1);
        MatVecMulAB(eta_tilde, V1, FI_2);
        MatVecMulAB(rho, omega_dot, V1);
        for (int i = 0; i < 3; i++) {
            FI_2(i) += V1(i);
        }
        MatMulAB(omega_tilde, rho, M1);
        MatVecMulAB(M1, omega, V1);
        for (int i = 0; i < 3; i++) {
            FI_2(i) += V1(i);
        }
    }
};

struct CalculateGravityForce {
    View_3::const_type gravity;
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3x3::const_type eta_tilde_;
    View_Nx3 v1_;
    View_Nx6 qp_FG_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto FG = Kokkos::subview(qp_FG_, i_qp, Kokkos::ALL);
        auto m = Muu(0, 0);
        VecScale(gravity, m, V1);
        for (int i = 0; i < 3; ++i) {
            FG(i) = V1(i);
        }
        MatVecMulAB(eta_tilde, V1, Kokkos::subview(FG, Kokkos::make_pair(3, 6)));
    }
};

struct CalculateOuu {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx3x3::const_type x0pupSS_;
    View_Nx3x3::const_type M_tilde_;
    View_Nx3x3::const_type N_tilde_;
    View_Nx6x6 qp_Ouu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M_tilde = Kokkos::subview(M_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Ouu = Kokkos::subview(qp_Ouu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto C21 = Kokkos::subview(Cuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
        for (int i = 0; i < Ouu.extent_int(0); ++i) {
            for (int j = 0; j < Ouu.extent_int(1); ++j) {
                Ouu(i, j) = 0.;
            }
        }
        auto Ouu_12 = Kokkos::subview(Ouu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        auto Ouu_22 = Kokkos::subview(Ouu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, Ouu_12);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Ouu_12(i, j) -= N_tilde(i, j);
            }
        }
        MatMulAB(C21, x0pupSS, Ouu_22);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Ouu_22(i, j) -= M_tilde(i, j);
            }
        }
    }
};

struct CalculatePuu {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx3x3::const_type x0pupSS_;
    View_Nx3x3::const_type N_tilde_;
    View_Nx6x6 qp_Puu_;
    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Puu = Kokkos::subview(qp_Puu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto C12 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        for (int i = 0; i < Puu.extent_int(0); ++i) {
            for (int j = 0; j < Puu.extent_int(1); ++j) {
                Puu(i, j) = 0.;
            }
        }
        auto Puu_21 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
        MatMulATB(x0pupSS, C11, Puu_21);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Puu_21(i, j) += N_tilde(i, j);
            }
        }
        auto Puu_22 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulATB(x0pupSS, C12, Puu_22);
    }
};

struct CalculateQuu {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx3x3::const_type x0pupSS_;
    View_Nx3x3::const_type N_tilde_;
    View_Nx3x3 M1_;
    View_Nx6x6 qp_Quu_;
    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Quu = Kokkos::subview(qp_Quu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        for (int i = 0; i < Quu.extent_int(0); ++i) {
            for (int j = 0; j < Quu.extent_int(1); ++j) {
                Quu(i, j) = 0.;
            }
        }
        auto Quu_22 = Kokkos::subview(Quu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M1(i, j) -= N_tilde(i, j);
            }
        }
        MatMulATB(x0pupSS, M1, Quu_22);
    }
};

struct CalculateGyroscopicMatrix {
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3::const_type qp_omega_;
    View_Nx3x3::const_type omega_tilde_;
    View_Nx3x3::const_type rho_;
    View_Nx3::const_type eta_;
    View_Nx3 v1_;
    View_Nx3 v2_;
    View_Nx3x3 M1_;
    View_Nx6x6 qp_Guu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
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
        for (int i = 0; i < Guu.extent_int(0); ++i) {
            for (int j = 0; j < Guu.extent_int(1); ++j) {
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
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Guu_12(i, j) += M1(j, i);
            }
        }
        // Guu_22 = omega.tilde() * rho - (rho * omega).tilde()
        auto Guu_22 = Kokkos::subview(Guu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(omega_tilde, rho, Guu_22);
        MatVecMulAB(rho, omega, V1);
        VecTilde(V1, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Guu_22(i, j) -= M1(i, j);
            }
        }
    }
};

struct CalculateInertiaStiffnessMatrix {
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3::const_type qp_u_ddot_;
    View_Nx3::const_type qp_omega_;
    View_Nx3::const_type qp_omega_dot_;
    View_Nx3x3::const_type omega_tilde_;
    View_Nx3x3::const_type omega_dot_tilde_;
    View_Nx3x3::const_type rho_;
    View_Nx3::const_type eta_;
    View_Nx3 v1_;
    View_Nx3x3 M1_;
    View_Nx3x3 M2_;
    View_Nx6x6 qp_Kuu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
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
        for (int i = 0; i < Kuu.extent_int(0); ++i) {
            for (int j = 0; j < Kuu.extent_int(1); ++j) {
                Kuu(i, j) = 0.;
            }
        }
        auto Kuu_12 = Kokkos::subview(Kuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        MatMulAB(omega_tilde, omega_tilde, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
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
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Kuu_22(i, j) += M1(i, j) - M2(i, j);
            }
        }
        MatMulAB(rho, omega_tilde, M1);
        MatVecMulAB(rho, omega, V1);
        VecTilde(V1, M2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M1(i, j) -= M2(i, j);
            }
        }
        MatMulAB(omega_tilde, M1, M2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Kuu_22(i, j) += M2(i, j);
            }
        }
    }
};

struct CalculateNodeForces_FE {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fc_;                                 //
    View_Nx6::const_type qp_Fd_;                                 //
    View_Nx6 node_FE_;                                           // Elastic force

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;

        // If node or qp indices don't apply to this element, return
        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FE = Kokkos::Array<double, 6>{};
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_c = weight * shape_deriv_(i, j_index);
            const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (int k = 0; k < 6; ++k) {
                local_FE[k] += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
            }
        }
        for (int k = 0; k < 6; ++k) {
            node_FE_(i, k) = local_FE[k];
        }
    }
};

struct CalculateNodeForces_FI {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fi_;                                 //
    View_Nx6 node_FI_;                                           // Elastic force

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;

        // If node or qp indices don't apply to this element, return
        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FI = Kokkos::Array<double, 6>{};
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_i = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (int k = 0; k < 6; ++k) {
                local_FI[k] += coeff_i * qp_Fi_(j, k);
            }
        }
        for (int k = 0; k < 6; ++k) {
            node_FI_(i, k) = local_FI[k];
        }
    }
};

struct CalculateNodeForces_FG {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fg_;                                 //
    View_Nx6 node_FG_;                                           // Elastic force

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;

        // If node or qp indices don't apply to this element, return
        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FG = Kokkos::Array<double, 6>{};
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_g = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (int k = 0; k < 6; ++k) {
                local_FG[k] += coeff_g * qp_Fg_(j, k);
            }
        }
        for (int k = 0; k < 6; ++k) {
            node_FG_(i, k) = local_FG[k];
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
    void operator()(const int i_elem) const {
        auto idx = elem_indices[i_elem];

        for (int i = idx.node_range.first; i < idx.node_range.second; ++i) {
            for (int j = 0; j < 6; ++j) {
                node_FE_(i, j) = 0.;
                node_FG_(i, j) = 0.;
                node_FI_(i, j) = 0.;
            }
        }

        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {    // Nodes
            for (int j_index = 0; j_index < idx.num_qps; ++j_index) {  // QPs
                const auto i = idx.node_range.first + i_index;
                const auto j = idx.qp_range.first + j_index;
                const auto weight = qp_weight_(j);
                const auto coeff_c = weight * shape_deriv_(i, j_index);
                const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
                const auto coeff_i = coeff_d;
                const auto coeff_g = coeff_d;
                for (int k = 0; k < 6; ++k) {  // Components
                    node_FE_(i, k) += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                }
                for (int k = 0; k < 6; ++k) {  // Components
                    node_FI_(i, k) += coeff_i * qp_Fi_(j, k);
                }
                for (int k = 0; k < 6; ++k) {  // Components
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
            [=](int i_index, int j) {
                const auto i = idx.node_range.first + i_index;
                node_FE_(i, j) = 0.;
                node_FG_(i, j) = 0.;
                node_FI_(i, j) = 0.;
            }
        );

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, idx.num_nodes), [=](int i_index) {
            for (int j_index = 0; j_index < idx.num_qps; ++j_index) {  // QPs
                const auto i = idx.node_range.first + i_index;
                const auto j = idx.qp_range.first + j_index;
                const auto weight = qp_weight_(j);
                const auto coeff_c = weight * shape_deriv_(i, j_index);
                const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
                const auto coeff_i = coeff_d;
                const auto coeff_g = coeff_d;
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 6), [=](int k) {
                    node_FE_(i, k) += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                });
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 6), [=](int k) {
                    node_FI_(i, k) += coeff_i * qp_Fi_(j, k);
                });
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 6), [=](int k) {
                    node_FG_(i, k) += coeff_g * qp_Fg_(j, k);
                });
            }
        });
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;

        // If node or qp indices don't apply to this element, return
        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FE = Kokkos::Array<double, 6>{};
        auto local_FI = Kokkos::Array<double, 6>{};
        auto local_FG = Kokkos::Array<double, 6>{};

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_c = weight * shape_deriv_(i, j_index);
            const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            const auto coeff_i = coeff_d;
            const auto coeff_g = coeff_d;
            for (int k = 0; k < 6; ++k) {
                local_FE[k] += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                local_FI[k] += coeff_i * qp_Fi_(j, k);
                local_FG[k] += coeff_g * qp_Fg_(j, k);
            }
        }

        for (int k = 0; k < 6; ++k) {
            node_FE_(i, k) = local_FE[k];
            node_FI_(i, k) = local_FI[k];
            node_FG_(i, k) = local_FG[k];
        }
    }
};

struct IntegrateMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<int*>::const_type node_state_indices;           // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6x6::const_type qp_M_;                                //
    View_NxN_atomic gbl_M_;                                      //

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        const auto idx = elem_indices[i_elem];

        for (int i = idx.node_range.first; i < idx.node_range.second; ++i) {      // Nodes
            for (int j = idx.node_range.first; j < idx.node_range.second; ++j) {  // Nodes
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (int k = 0; k < idx.num_qps; ++k) {
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto coeff = w * phi_i * phi_j * jacobian;
                    for (int m = 0; m < kLieAlgebraComponents; ++m) {
                        for (int n = 0; n < kLieAlgebraComponents; ++n) {
                            local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                        }
                    }
                }
                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                for (int m = 0; m < kLieAlgebraComponents; ++m) {
                    for (int n = 0; n < kLieAlgebraComponents; ++n) {
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
            [=](int i_index, int j_index) {
                const auto i = i_index + idx.node_range.first;
                const auto j = j_index + idx.node_range.first;
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (int k = 0; k < idx.num_qps; ++k) {
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto coeff = w * phi_i * phi_j * jacobian;
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorMDRange(member, 6, 6),
                        [=](int m, int n) {
                            local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                        }
                    );
                }
                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                Kokkos::parallel_for(Kokkos::ThreadVectorMDRange(member, 6, 6), [=](int m, int n) {
                    gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                });
            }
        );
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index, const int j_index) const {
        const auto idx = elem_indices[i_elem];

        // If node or qp indices don't apply to this element, return
        if (i_index >= idx.num_nodes || j_index >= idx.num_nodes) {
            return;
        }

        const auto i = i_index + idx.node_range.first;
        const auto j = j_index + idx.node_range.first;
        auto local_M_data = Kokkos::Array<double, 36>{};
        auto local_M =
            Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(local_M_data.data());
        for (int k = 0; k < idx.num_qps; ++k) {
            const auto k_qp = idx.qp_range.first + k;
            const auto w = qp_weight_(k_qp);
            const auto jacobian = qp_jacobian_(k_qp);
            const auto phi_i = shape_interp_(i, k);
            const auto phi_j = shape_interp_(j, k);
            const auto coeff = w * phi_i * phi_j * jacobian;
            for (int m = 0; m < kLieAlgebraComponents; ++m) {
                for (int n = 0; n < kLieAlgebraComponents; ++n) {
                    local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                }
            }
        }
        const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
        const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
        for (int m = 0; m < kLieAlgebraComponents; ++m) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index, const int j_index, const int k) const {
        const auto idx = elem_indices[i_elem];

        // If node or qp indices don't apply to this element, return
        if (i_index >= idx.num_nodes || j_index >= idx.num_nodes || k >= idx.num_qps) {
            return;
        }

        const auto i = i_index + idx.node_range.first;
        const auto j = j_index + idx.node_range.first;
        auto local_M_data = Kokkos::Array<double, 36>{};
        auto local_M =
            Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(local_M_data.data());
        const auto k_qp = idx.qp_range.first + k;
        const auto w = qp_weight_(k_qp);
        const auto jacobian = qp_jacobian_(k_qp);
        const auto phi_i = shape_interp_(i, k);
        const auto phi_j = shape_interp_(j, k);
        const auto coeff = w * phi_i * phi_j * jacobian;
        for (int m = 0; m < kLieAlgebraComponents; ++m) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                local_M(m, n) += coeff * qp_M_(k_qp, m, n);
            }
        }
        const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
        const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
        for (int m = 0; m < kLieAlgebraComponents; ++m) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
            }
        }
    }
};

struct IntegrateElasticStiffnessMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<int*>::const_type node_state_indices;           // Element indices
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
    void operator()(const int i_elem) const {
        const auto idx = elem_indices[i_elem];

        for (int i = idx.node_range.first; i < idx.node_range.second; ++i) {      // Nodes
            for (int j = idx.node_range.first; j < idx.node_range.second; ++j) {  // Nodes
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (int k = 0; k < idx.num_qps; ++k) {  // QPs
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
                    for (int m = 0; m < 6; ++m) {      // Matrix components
                        for (int n = 0; n < 6; ++n) {  // Matrix components
                            local_M(m, n) +=
                                coeff_P * qp_Puu_(k_qp, m, n) + coeff_Q * qp_Quu_(k_qp, m, n) +
                                coeff_C * qp_Cuu_(k_qp, m, n) + coeff_O * qp_Ouu_(k_qp, m, n);
                        }
                    }
                }

                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                for (int m = 0; m < 6; ++m) {
                    for (int n = 0; n < 6; ++n) {
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
            [=](int i_index, int j_index) {
                const auto i = i_index + idx.node_range.first;
                const auto j = j_index + idx.node_range.first;
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (int k = 0; k < idx.num_qps; ++k) {
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
                        [=](int m, int n) {
                            local_M(m, n) +=
                                coeff_P * qp_Puu_(k_qp, m, n) + coeff_Q * qp_Quu_(k_qp, m, n) +
                                coeff_C * qp_Cuu_(k_qp, m, n) + coeff_O * qp_Ouu_(k_qp, m, n);
                        }
                    );
                }

                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                Kokkos::parallel_for(Kokkos::ThreadVectorMDRange(member, 6, 6), [=](int m, int n) {
                    gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                });
            }
        );
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index, const int j_index) const {
        const auto idx = elem_indices[i_elem];

        // If node or qp indices don't apply to this element, return
        if (i_index >= idx.num_nodes || j_index >= idx.num_nodes) {
            return;
        }

        const auto i = i_index + idx.node_range.first;
        const auto j = j_index + idx.node_range.first;
        auto local_M_data = Kokkos::Array<double, 36>{};
        auto local_M =
            Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(local_M_data.data());
        for (int k = 0; k < idx.num_qps; ++k) {
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
            for (int m = 0; m < 6; ++m) {      // Matrix components
                for (int n = 0; n < 6; ++n) {  // Matrix components
                    local_M(m, n) += coeff_P * qp_Puu_(k_qp, m, n) + coeff_Q * qp_Quu_(k_qp, m, n) +
                                     coeff_C * qp_Cuu_(k_qp, m, n) + coeff_O * qp_Ouu_(k_qp, m, n);
                }
            }
        }

        const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
        const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
        for (int m = 0; m < 6; ++m) {
            for (int n = 0; n < 6; ++n) {
                gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index, const int j_index, const int k) const {
        const auto idx = elem_indices(i_elem);

        if (i_index >= idx.num_nodes || j_index >= idx.num_nodes || k >= idx.num_qps) {
            return;
        }

        const auto i = i_index + idx.node_range.first;
        const auto j = j_index + idx.node_range.first;
        auto local_M_data = Kokkos::Array<double, 36>{};
        auto local_M =
            Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(local_M_data.data());
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
        for (int m = 0; m < 6; ++m) {      // Matrix components
            for (int n = 0; n < 6; ++n) {  // Matrix components
                local_M(m, n) = coeff_P * qp_Puu_(k_qp, m, n) + coeff_Q * qp_Quu_(k_qp, m, n) +
                                coeff_C * qp_Cuu_(k_qp, m, n) + coeff_O * qp_Ouu_(k_qp, m, n);
            }
        }

        const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
        const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
        for (int m = 0; m < 6; ++m) {
            for (int n = 0; n < 6; ++n) {
                gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
            }
        }
    }
};

struct IntegrateResidualVector {
    Kokkos::View<int*>::const_type node_state_indices_;
    View_Nx6::const_type node_FE_;  // Elastic force
    View_Nx6::const_type node_FI_;  // Inertial force
    View_Nx6::const_type node_FG_;  // Gravity force
    View_Nx6::const_type node_FX_;  // External force
    View_N_atomic residual_vector_;

    KOKKOS_INLINE_FUNCTION void operator()(const int i_node) const {
        auto i_rv_start = node_state_indices_(i_node) * kLieAlgebraComponents;
        for (int j = 0; j < 6; j++) {
            residual_vector_(i_rv_start + j) += node_FE_(i_node, j) + node_FI_(i_node, j) -
                                                node_FX_(i_node, j) - node_FG_(i_node, j);
        }
    }
};

}  // namespace openturbine
