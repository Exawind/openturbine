#include "solver.hpp"

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

void PredictNextState(
    State& state, double h, double beta, double gamma, double alpha_m, double alpha_f
) {
    Kokkos::deep_copy(state.lambda, 0.);
    Kokkos::deep_copy(state.q_prev, state.q);

    // Predict the new state values
    Kokkos::parallel_for(
        "Loop1", state.q.extent(0),
        KOKKOS_LAMBDA(const size_t i) {
            for (size_t j = 0; j < kLieAlgebraComponents; ++j) {
                double v = state.v(i, j);
                double vd = state.vd(i, j);
                double a = state.a(i, j);
                state.vd(i, j) = 0.;
                state.a(i, j) = (alpha_f * vd - alpha_m * a) / (1. - alpha_m);
                state.v(i, j) = v + h * (1. - gamma) * a + gamma * h * state.a(i, j);
                state.q_delta(i, j) = v + (0.5 - beta) * h * a + beta * h * state.a(i, j);
            }
        }
    );

    // Update q
    Kokkos::parallel_for(
        "UpdateQ", state.q.extent(0), CalculateQ{h, state.q_delta, state.q_prev, state.q}
    );
}

// void UpdateDelta(State& state, double h, double beta_prime, double gamma_prime) {
// let x_delta = delta.columns(0, self.num_system_nodes);
// self.q_delta += x_delta / h;
// self.v += gamma_prime * x_delta;
// self.vd += beta_prime * x_delta;
// self.update_q(h);
// let lambda_delta = delta.columns(self.num_system_nodes, self.num_constraint_nodes);
// self.lambda -= lambda_delta;
// }

}  // namespace oturb