#pragma once

#include "types.hpp"

namespace openturbine {

//------------------------------------------------------------------------------
// Functor helper functions which should be reworked later
//------------------------------------------------------------------------------

KOKKOS_FUNCTION
void VectorTilde(double a, double v[3], double m[3][3]) {
    m[0][0] = 0.;
    m[0][1] = -v[2] * a;
    m[0][2] = v[1] * a;
    m[1][0] = v[2] * a;
    m[1][1] = 0.;
    m[1][2] = -v[0] * a;
    m[2][0] = -v[1] * a;
    m[2][1] = v[0] * a;
    m[2][2] = 0.;
}

KOKKOS_FUNCTION
void Mat3xMat3(double m1[3][3], double m2[3][3], double m3[3][3]) {
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m3[i][j] = 0.;
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                m3[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

KOKKOS_FUNCTION
void RotationVectorToQuaternion(double phi[3], double quaternion[4]) {
    double angle = std::sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);

    // Convert rotation vector to quaternion
    if (Kokkos::abs(angle) < 1e-12) {
        quaternion[0] = 1.0;
        quaternion[1] = 0.0;
        quaternion[2] = 0.0;
        quaternion[3] = 0.0;
    } else {
        double sin_angle = Kokkos::sin(angle / 2.0);
        double cos_angle = Kokkos::cos(angle / 2.0);
        double factor = sin_angle / angle;
        quaternion[0] = cos_angle;
        quaternion[1] = phi[0] * factor;
        quaternion[2] = phi[1] * factor;
        quaternion[3] = phi[2] * factor;
    }
}

//------------------------------------------------------------------------------
// Functors
//------------------------------------------------------------------------------

struct CalculateNextState {
    double h;
    double alpha_f;
    double alpha_m;
    double beta;
    double gamma;
    View_Nx6 q_delta;
    View_Nx6 v;
    View_Nx6 vd;
    View_Nx6 a;

    KOKKOS_FUNCTION
    void operator()(const size_t i) const {
        for (size_t j = 0; j < kLieAlgebraComponents; ++j) {
            double v_p = v(i, j);    // Save velocity from previous iteration
            double vd_p = vd(i, j);  // Save acceleration from previous iteration
            double a_p = a(i, j);    // Save algorithmic acceleration from previous iteration
            vd(i, j) = 0.;
            a(i, j) = (alpha_f * vd_p - alpha_m * a_p) / (1. - alpha_m);
            v(i, j) = v_p + h * (1. - gamma) * a_p + gamma * h * a(i, j);
            q_delta(i, j) = v_p + (0.5 - beta) * h * a_p + beta * h * a(i, j);
        }
    }
};

template <typename Subview_NxN>
struct UpdateIterationMatrix {
    Subview_NxN St_12;
    View_NxN B;
    size_t num_constraint_dofs;
    size_t num_system_dofs;

    KOKKOS_FUNCTION
    void operator()(size_t) const {
        for (size_t i = 0; i < num_constraint_dofs; ++i) {
            for (size_t j = 0; j < num_system_dofs; ++j) {
                St_12(j, i) = B(i, j);
            }
        }
    }
};

struct UpdateStaticPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    View_N x_delta;
    View_Nx6 q_delta;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        for (size_t j = 0; j < kLieAlgebraComponents; j++) {
            double delta = x_delta(i_node * 6 + j);
            q_delta(i_node, j) += delta / h;
        }
    }
};

struct UpdateDynamicPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    View_N x_delta;
    View_Nx6 q_delta;
    View_Nx6 v;
    View_Nx6 vd;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        for (size_t j = 0; j < kLieAlgebraComponents; j++) {
            double delta = x_delta(i_node * 6 + j);
            q_delta(i_node, j) += delta / h;
            v(i_node, j) += gamma_prime * delta;
            vd(i_node, j) += beta_prime * delta;
        }
    }
};

struct UpdateLambdaPrediction {
    View_N lambda_delta;
    View_N lambda;

    KOKKOS_FUNCTION
    void operator()(const size_t i_lambda) const { lambda(i_lambda) -= lambda_delta(i_lambda); }
};

struct CalculateTangentOperator {
    double h;
    View_Nx6 q_delta;
    View_NxN T;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        size_t j = i_node * kLieAlgebraComponents;
        for (size_t k = 0; k < kLieAlgebraComponents; ++k) {
            T(j + k, j + k) = 1.0;
        }
        // Rotation vector
        double rv[3] = {h * q_delta(i_node, 3), h * q_delta(i_node, 4), h * q_delta(i_node, 5)};
        double phi = Kokkos::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
        if (phi > 1.0e-16) {
            j += 3;
            double m1[3][3], m2[3][3], m3[3][3], m4[3][3];
            double tmp1 = (Kokkos::cos(phi) - 1.) / (phi * phi);
            double tmp2 = (1. - Kokkos::sin(phi) / phi) / (phi * phi);
            VectorTilde(tmp1, rv, m1);
            VectorTilde(tmp2, rv, m2);
            VectorTilde(1.0, rv, m3);
            Mat3xMat3(m2, m3, m4);
            for (size_t k = 0; k < 3; ++k) {
                for (size_t n = 0; n < 3; ++n) {
                    T(j + k, j + n) += m1[k][n] + m4[k][n];
                }
            }
        }
    }
};

struct CalculateDisplacement {
    double h;
    View_Nx6 q_delta;
    View_Nx7 q_prev;
    View_Nx7 q;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        // Calculate new displacements
        q(i_node, 0) = q_prev(i_node, 0) + h * q_delta(i_node, 0);
        q(i_node, 1) = q_prev(i_node, 1) + h * q_delta(i_node, 1);
        q(i_node, 2) = q_prev(i_node, 2) + h * q_delta(i_node, 2);

        // Delta rotation vector to quaternion
        auto quat_delta = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(
            h * q_delta(i_node, 3), h * q_delta(i_node, 4), h * q_delta(i_node, 5)
        );

        // Previous rotation as quaternion
        Quaternion quat_prev(
            q_prev(i_node, 3), q_prev(i_node, 4), q_prev(i_node, 5), q_prev(i_node, 6)
        );

        // Compose delta rotation and previous rotation
        auto quat_new = quat_delta * quat_prev;

        // Set new values of rotation quaternion
        q(i_node, 3) = quat_new.GetScalarComponent();
        q(i_node, 4) = quat_new.GetXComponent();
        q(i_node, 5) = quat_new.GetYComponent();
        q(i_node, 6) = quat_new.GetZComponent();
    }
};

struct CalculateConstraintX0 {
    Kokkos::View<Constraints::NodeIndices*> node_indices;
    View_Nx7 node_x0;
    View_Nx3 constraint_X0;

    KOKKOS_FUNCTION
    void operator()(const size_t i_constraint) const {
        auto i_node1 = node_indices(i_constraint).base_node_index;
        auto i_node2 = node_indices(i_constraint).constrained_node_index;

        if (i_node1 == (size_t)-1) {
            constraint_X0(i_constraint, 0) = node_x0(i_node2, 0);
            constraint_X0(i_constraint, 1) = node_x0(i_node2, 1);
            constraint_X0(i_constraint, 2) = node_x0(i_node2, 2);
        } else {
            constraint_X0(i_constraint, 0) = node_x0(i_node2, 0) - node_x0(i_node1, 0);
            constraint_X0(i_constraint, 1) = node_x0(i_node2, 1) - node_x0(i_node1, 1);
            constraint_X0(i_constraint, 2) = node_x0(i_node2, 2) - node_x0(i_node1, 2);
        }
    }
};

struct CalculateConstraintResidualGradient {
    Kokkos::View<Constraints::NodeIndices*> node_indices;
    View_Nx3 constraint_X0;
    View_Nx7 constraint_u;
    View_Nx7 node_u;
    View_N Phi;
    View_NxN B;

    KOKKOS_FUNCTION
    void operator()(const size_t i_constraint) const {
        auto i_node1 = node_indices(i_constraint).base_node_index;
        auto i_node2 = node_indices(i_constraint).constrained_node_index;

        Quaternion R1;
        if (i_node1 == (size_t)-1) {
            R1 = Quaternion(
                constraint_u(i_constraint, 3), constraint_u(i_constraint, 4),
                constraint_u(i_constraint, 5), constraint_u(i_constraint, 6)
            );
        } else {
            R1 = Quaternion(
                node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)
            );
        }

        Quaternion R2(
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)
        );

        Vector X0(
            constraint_X0(i_constraint, 0), constraint_X0(i_constraint, 1),
            constraint_X0(i_constraint, 2)
        );

        Vector u2(node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2));

        auto i_row = i_constraint * kLieAlgebraComponents;
        auto i_col = i_node2 * kLieAlgebraComponents;

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        auto Phi_x = u2 + X0 - R1 * X0;
        auto Phi_p = (R2 * R1.GetInverse()).Axial();

        // Displacement residual
        Phi(i_row + 0) = Phi_x.GetXComponent();
        Phi(i_row + 1) = Phi_x.GetYComponent();
        Phi(i_row + 2) = Phi_x.GetZComponent();

        // Rotation residual
        Phi(i_row + 3) = Phi_p.GetXComponent();
        Phi(i_row + 4) = Phi_p.GetYComponent();
        Phi(i_row + 5) = Phi_p.GetZComponent();

        //----------------------------------------------------------------------
        // Gradient matrix
        //----------------------------------------------------------------------

        auto R2R1T = (R1 * R2.GetInverse());
        auto mR2R1T = R2R1T.to_rotation_matrix();
        auto tr = R2R1T.Trace();
        RotationMatrix trI3(tr, 0., 0., 0., tr, 0., 0., 0., tr);

        // Displacement gradient (identity)
        for (size_t i = 0; i < 3; ++i) {
            B(i_row + i, i_col + i) = 1.;
        }

        // Rotation gradient
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                B(i_row + i + 3, i_col + j + 3) = (trI3(i, j) - mR2R1T(i, j)) / 2.;
            }
        }
    }
};

struct PreconditionSt {
    View_NxN St;
    double conditioner;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t j) const { St(i, j) *= conditioner; }
};

struct PostconditionSt {
    View_NxN St;
    double conditioner;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t j) const { St(i, j) /= conditioner; }
};

struct ConditionR {
    View_N R;
    double conditioner;

    KOKKOS_FUNCTION
    void operator()(size_t i) const { R(i) *= conditioner; }
};

struct ConditionSystem {
    size_t num_system_dofs;
    size_t num_dofs;
    double conditioner;
    View_NxN St;
    View_N R;

    KOKKOS_FUNCTION
    void operator()(const size_t) const {
        // DL = (i < num_system_dofs) ? conditioner : 1.0
        // DR = (i >= num_system_dofs) ? 1.0 / conditioner : 1.0

        // DL * St * DR
        // Premultiplying by diagonal matrix DL matrix effectively multiplies
        // each row by the diagonal element. The diagonal element is conditioner
        // for the system dofs and 1.0 for the constraint dofs
        for (size_t i = 0; i < num_system_dofs; ++i) {
            for (size_t j = 0; j < num_dofs; ++j) {
                St(i, j) *= conditioner;
            }
        }
        // Postmultiplying by diagonal matrix DR matrix effectively multiplies
        // each column by the diagonal element. The diagonal element is 1.0
        // for the system dofs and 1/conditioner for the constraint dofs
        for (size_t i = 0; i < num_dofs; ++i) {
            for (size_t j = num_system_dofs; j < num_dofs; ++j) {
                St(i, j) /= conditioner;
            }
        }

        // R * DL
        // DL is conditioner for system dofs, 1.0 for constraint dofs
        for (size_t i = 0; i < num_system_dofs; ++i) {
            R(i) *= conditioner;
        }
    }
};

struct UnconditionSolution {
    size_t num_system_dofs;
    double conditioner;
    View_N x;

    KOKKOS_FUNCTION
    void operator()(const size_t i) const {
        // DR = (i > num_system_dofs) ? 1.0 / conditioner : 1.0
        // x * DR
        if (i >= num_system_dofs) {
            x(i) /= conditioner;
        }
    }
};

struct CalculateErrorSumSquares {
    using value_type = double;

    double atol;
    double rtol;
    View_Nx6 q_delta;
    View_N x;

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i, double& err) const {
        err += Kokkos::pow(x(i) / (atol + rtol * Kokkos::abs(q_delta(i / 6, i % 6))), 2.);
    }
};

struct UpdateAlgorithmicAcceleration {
    View_Nx6 acceleration;
    View_Nx6 vd;
    double alpha_f;
    double alpha_m;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        for (size_t j = 0; j < 6; ++j) {
            acceleration(i, j) += (1. - alpha_f) / (1. - alpha_m) * vd(i, j);
        }
    }
};

}  // namespace openturbine
