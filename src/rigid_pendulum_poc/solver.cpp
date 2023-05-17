#include "src/rigid_pendulum_poc/solver.h"

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

GeneralizedAlphaSolver::GeneralizedAlphaSolver(
    std::unique_ptr<LinearSystemSolver> linear_system_solver)
    : linear_system_solver_(std::move(linear_system_solver)) {
}

void GeneralizedAlphaSolver::AlphaStep(const Eigen::VectorXd& gen_coords,
                                       const Eigen::VectorXd& gen_coords_dot,
                                       const Eigen::VectorXd& gen_coords_ddot,
                                       const Eigen::VectorXd& accelaration) {
    auto log = util::Log::Get();
    log->Debug("AlphaStep\n");

    // Some placeholders
    float step_size = 0.1;
    float alpha_f = 0.5;
    float alpha_m = 0.5;
    float beta = 0.25;
    float gamma = 0.5;
    int max_iterations = 10;

    Eigen::VectorXd gen_coords_next = gen_coords + step_size * gen_coords_dot +
                                      step_size * step_size * (0.5 - beta) * accelaration;
    Eigen::VectorXd gen_coords_dot_next = gen_coords_dot + step_size * (1 - gamma) * accelaration;
    Eigen::VectorXd lambda_next {};
    Eigen::VectorXd accelaration_next =
        (1 / (1 - alpha_m)) * (alpha_f * gen_coords_ddot - alpha_m * accelaration);
    gen_coords_next = gen_coords_next + step_size * step_size * beta * accelaration_next;
    gen_coords_dot_next = gen_coords_dot_next + step_size * beta * accelaration_next;
    Eigen::VectorXd gen_coords_ddot_next {};

    for (int i = 0; i < max_iterations; i++) {
        // Compute the residuals
        // TODO Define a function to compute the residuals
        Eigen::VectorXd r1 = gen_coords_next - gen_coords - step_size * gen_coords_dot -
                             step_size * step_size * (0.5 - beta) * accelaration;
        Eigen::VectorXd r2 =
            gen_coords_dot_next - gen_coords_dot - step_size * (1 - gamma) * accelaration;

        // Compute the jacobian
        Eigen::MatrixXd jacobian {};
        // TODO Define a function to compute the jacobian
        // jacobian = jacobian - step_size* step_size* (0.5 - beta) * alpha_f;

        // Compute the iteration matrix
        // TODO Define a function to compute the iteration matrix
        Eigen::MatrixXd iteration_matrix {};
        // iteration_matrix = iteration_matrix - step_size* step_size* (0.5 - beta) * alpha_m;

        // gen_coords[i] = gen_coords_next[i];
        // gen_coords_dot[i] = gen_coords_dot_next[i];
        // gen_coords_ddot[i] = gen_coords_ddot_next[i];
        // accelaration[i] = accelaration_next[i];
    }
}

}  // namespace openturbine::rigid_pendulum
