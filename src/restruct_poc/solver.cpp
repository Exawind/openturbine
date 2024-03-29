#include "solver.hpp"

#include "beams.hpp"

namespace oturb {

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

/// Multiplies provided quaternion with this quaternion and returns the result
KOKKOS_INLINE_FUNCTION
void QuaternionCompose(double q1[4], double q2[4], double qn[4]) {
    qn[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    qn[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    qn[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    qn[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
}

struct PredictNextState {
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
            double v_p = v(i, j);    // Save value from previous iteration
            double vd_p = vd(i, j);  // Save value from previous iteration
            double a_p = a(i, j);    // Save value from previous iteration
            vd(i, j) = 0.;
            a(i, j) = (alpha_f * vd_p - alpha_m * a_p) / (1. - alpha_m);
            v(i, j) = v_p + h * (1. - gamma) * a_p + gamma * h * a(i, j);
            q_delta(i, j) = v_p + (0.5 - beta) * h * a_p + beta * h * a(i, j);
        }
    }
};

struct CalculateQ {
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

        // Rotation vector
        double phi[3] = {h * q_delta(i_node, 3), h * q_delta(i_node, 4), h * q_delta(i_node, 5)};

        // Compose quaternions
        double quat_delta[4], quat_prev[4], quat_new[4];
        RotationVectorToQuaternion(phi, quat_delta);
        quat_prev[0] = q_prev(i_node, 3);
        quat_prev[1] = q_prev(i_node, 4);
        quat_prev[2] = q_prev(i_node, 5);
        quat_prev[3] = q_prev(i_node, 6);
        QuaternionCompose(quat_delta, quat_prev, quat_new);

        // Update q
        q(i_node, 3) = quat_new[0];
        q(i_node, 4) = quat_new[1];
        q(i_node, 5) = quat_new[2];
        q(i_node, 6) = quat_new[3];
    }
};

void PredictNextState(Solver& solver) {
    Kokkos::deep_copy(solver.state.lambda, 0.);
    Kokkos::deep_copy(solver.state.q_prev, solver.state.q);

    // Predict the new state values
    Kokkos::parallel_for(
        "PredictNextState", solver.num_system_nodes,
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < kLieAlgebraComponents; ++j) {
                double v = solver.state.v(i, j);
                double vd = solver.state.vd(i, j);
                double a = solver.state.a(i, j);
                solver.state.vd(i, j) = 0.;
                solver.state.a(i, j) =
                    (solver.alpha_f * vd - solver.alpha_m * a) / (1. - solver.alpha_m);
                solver.state.v(i, j) = v + solver.h * (1. - solver.gamma) * a +
                                       solver.gamma * solver.h * solver.state.a(i, j);
                solver.state.q_delta(i, j) = v + (0.5 - solver.beta) * solver.h * a +
                                             solver.beta * solver.h * solver.state.a(i, j);
            }
        }
    );

    // Update q
    Kokkos::parallel_for(
        "UpdateQ", solver.state.q.extent(0),
        CalculateQ{solver.h, solver.state.q_delta, solver.state.q_prev, solver.state.q}
    );
}

struct UpdateDeltaDynamic {
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

struct UpdateDeltaStatic {
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

struct UpdateDeltaLambda {
    View_N lambda_delta;
    View_N lambda;

    KOKKOS_FUNCTION
    void operator()(const size_t i_lambda) const { lambda(i_lambda) -= lambda_delta(i_lambda); }
};

void UpdateDelta(Solver& solver, double h, double beta_prime, double gamma_prime) {
    // Get delta x subview of solution vector
    auto x_delta = Kokkos::subview(solver.x, Kokkos::make_pair((size_t)0, solver.num_system_dofs));

    if (solver.is_dynamic_solve) {
        // Calculate change in state based on dynamic solution iteration
        Kokkos::parallel_for(
            "UpdateDeltaDynamic", solver.num_system_nodes,
            UpdateDeltaDynamic{
                h, beta_prime, gamma_prime, x_delta, solver.state.q_delta, solver.state.v,
                solver.state.vd}
        );
    } else {
        Kokkos::parallel_for(
            // Calculate change in state based on static solution iteration
            "UpdateDeltaStatic", solver.num_system_nodes,
            UpdateDeltaStatic{h, beta_prime, gamma_prime, x_delta, solver.state.q_delta}
        );
    }

    // If solver is using constraints
    if (solver.num_constraint_dofs > 0) {
        // Get delta lambda subview of solution vector
        auto lambda_delta =
            Kokkos::subview(solver.x, Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs));
        // Update lambda in state
        Kokkos::parallel_for(
            "UpdateDeltaLambda", solver.num_system_nodes,
            UpdateDeltaLambda{lambda_delta, solver.state.lambda}
        );
    }
}

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
        double rv[3] = {q_delta(i_node, 3), q_delta(i_node, 4), q_delta(i_node, 5)};
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
                    T(j + k, j + n) += m1[k][n] + m2[k][n];
                }
            }
        }
    }
};

struct CalculateConstraintX0 {
    Kokkos::View<size_t* [2]> node_indices;
    View_Nx7 node_x0;
    View_Nx3 constraint_X0;

    KOKKOS_FUNCTION
    void operator()(const size_t i_constraint) const {
        auto i_node1 = node_indices(i_constraint, 0);  // Reference node index
        auto i_node2 = node_indices(i_constraint, 1);  // Constrained node index

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

// struct CalculateConstraintResidual {
//     Kokkos::View<size_t* [2]> node_indices;
//     View_Nx3 constraint_X0;
//     View_Nx7 node_u;
//     View_Nx7 constraint_u;
//     View_N Phi;

//     KOKKOS_FUNCTION
//     void operator()(const size_t i_constraint) const {
//         // Node indices
//         auto i_node1 = node_indices(i_constraint, 0);  // Reference node index
//         auto i_node2 = node_indices(i_constraint, 1);  // Constrained node index

//         // auto i_Phi = i_node2 * kLieAlgebraComponents;
//         if (i_node1 == (size_t)-1) {
//             // Displacement
//             Phi(i_Phi + 0) = node_x0(i_node2, 0);
//             Phi(i_Phi + 1) = node_x0(i_node2, 1);
//             Phi(i_Phi + 2) = node_x0(i_node2, 2);
//             // Rotation
//             Phi(i_Phi + 3) = node_x0(i_node2, 0);
//             Phi(i_Phi + 4) = node_x0(i_node2, 1);
//             Phi(i_Phi + 5) = node_x0(i_node2, 2);
//         } else {
//             // TODO
//             Phi(i_Phi, 0);
//             Phi(i_Phi, 1);
//             Phi(i_Phi, 2);
//         }
//     }
// };

void InitializeConstraints(Solver& solver, Beams& beams) {
    Kokkos::parallel_for(
        "CalculateConstraintX0", solver.num_constraint_nodes,
        CalculateConstraintX0{solver.constraints.node_indices, beams.node_x0, solver.constraints.X0}
    );
}

void Step(Solver& solver, Beams& beams) {
    // Predict state at end of step
    PredictNextState(solver);

    auto system_range = Kokkos::make_pair((size_t)0, solver.num_system_dofs);
    auto constraint_range = Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs);

    auto system_residual = Kokkos::subview(solver.R, system_range);
    auto lambda_residual = Kokkos::subview(solver.R, constraint_range);

    auto St_11 = Kokkos::subview(solver.St, system_range, system_range);
    auto St_12 = Kokkos::subview(solver.St, system_range, constraint_range);
    auto St_21 = Kokkos::subview(solver.St, constraint_range, system_range);

    // Perform convergence iterations
    for (size_t iter = 0; iter < solver.max_iter; ++iter) {
        // Update beam elements state from solvers
        UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);

        // Tangent operator
        Kokkos::deep_copy(solver.T, 0.);
        Kokkos::parallel_for(
            "TangentOperator", solver.num_system_nodes,
            CalculateTangentOperator{solver.h, solver.state.q_delta, solver.T}
        );

        // Assemble residual vector
        Kokkos::deep_copy(solver.R, 0.);
        AssembleResidualVector(beams, system_residual);

        // If dynamic solution
        if (solver.is_dynamic_solve) {
            Kokkos::deep_copy(solver.M, 0.);
            Kokkos::deep_copy(solver.G, 0.);
            AssembleMassMatrix(beams, solver.M);
            AssembleGyroscopicInertiaMatrix(beams, solver.G);
        }

        // Assemble matrices from beam elements
        Kokkos::deep_copy(solver.K, 0.);
        AssembleElasticStiffnessMatrix(beams, solver.K);
        AssembleInertialStiffnessMatrix(beams, solver.K);

        // Constraints
        if (solver.num_constraint_dofs > 0) {
            Kokkos::deep_copy(solver.constraints.B, 0.);
            Kokkos::deep_copy(solver.constraints.Phi, 0.);
        }

        // Iteration matrix
        Kokkos::deep_copy(solver.St, 0.);
        KokkosBlas::axpy(solver.beta_prime, solver.M, St_11);
        KokkosBlas::axpy(solver.gamma_prime, solver.G, St_11);
        // KokkosBlas::axpy(solver.K, solver.T, St_11);
    }
}

}  // namespace oturb