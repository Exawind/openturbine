#include "src/rigid_pendulum_poc/generalized_alpha_time_integrator.h"

#include "src/rigid_pendulum_poc/quaternion.h"
#include "src/rigid_pendulum_poc/solver.h"
#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double alpha_f, double alpha_m, double beta, double gamma, TimeStepper time_stepper
)
    : kALPHA_F_(alpha_f),
      kALPHA_M_(alpha_m),
      kBETA_(beta),
      kGAMMA_(gamma),
      time_stepper_(std::move(time_stepper)) {
    if (this->kALPHA_F_ < 0 || this->kALPHA_F_ > 1) {
        throw std::invalid_argument("Invalid value for alpha_f");
    }

    if (this->kALPHA_M_ < 0 || this->kALPHA_M_ > 1) {
        throw std::invalid_argument("Invalid value for alpha_m");
    }

    if (this->kBETA_ < 0 || this->kBETA_ > 0.50) {
        throw std::invalid_argument("Invalid value for beta");
    }

    if (this->kGAMMA_ < 0 || this->kGAMMA_ > 1) {
        throw std::invalid_argument("Invalid value for gamma");
    }

    this->is_converged_ = false;
}

std::vector<State> GeneralizedAlphaTimeIntegrator::Integrate(
    const State& initial_state, std::function<HostView2D(size_t)> matrix,
    std::function<HostView1D(size_t)> vector
) {
    auto log = util::Log::Get();

    std::vector<State> states{initial_state};
    auto n_steps = this->time_stepper_.GetNumberOfSteps();
    for (size_t i = 0; i < n_steps; i++) {
        log->Info("Integrating step number " + std::to_string(i + 1) + "\n");
        states.emplace_back(std::get<0>(this->AlphaStep(states[i], matrix, vector)));
        this->time_stepper_.AdvanceTimeStep();
    }

    log->Info("Time integration completed successfully!\n");

    return states;
}

std::tuple<State, HostView1D> GeneralizedAlphaTimeIntegrator::AlphaStep(
    const State& state, std::function<HostView2D(size_t)> matrix,
    std::function<HostView1D(size_t)> vector
) {
    auto gen_coords = state.GetGeneralizedCoordinates();
    auto velocity = state.GetVelocity();
    auto acceleration = state.GetAcceleration();
    auto algo_accleration = state.GetAlgorithmicAcceleration();

    // Initialize the updated algorithmic acceleration and an intermediate vector to assist
    // in updating the State
    auto algo_accleration_next =
        HostView1D("algorithmic_acceleration_next", algo_accleration.size());
    auto x = HostView1D("increment", velocity.size());

    // TODO: Provide actual constraints
    auto constraints = HostView1D("constraints", velocity.size());

    // Perform the linear update of the generalized alpha algorithm
    const auto h = this->time_stepper_.GetTimeStep();
    const auto size = velocity.size();
    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const int i) {
            acceleration(i) = 0.;
            algo_accleration_next(i) =
                (kALPHA_F_ * acceleration(i) - kALPHA_M_ * algo_accleration(i)) / (1. - kALPHA_M_);
            velocity(i) +=
                h * (1 - kGAMMA_) * algo_accleration(i) + h * kGAMMA_ * algo_accleration_next(i);
            x(i) = h * velocity(i) + h * h * (0.5 - kBETA_) * algo_accleration(i) +
                   h * h * kBETA_ * algo_accleration_next(i);
        }
    );

    const auto BETA_PRIME = (1 - kALPHA_M_) / (h * h * kBETA_ * (1 - kALPHA_F_));
    const auto GAMMA_PRIME = kGAMMA_ / (h * kBETA_);

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha algorithm
    auto log = util::Log::Get();
    log->Info(
        "Performing Newton-Raphson iterations to update the nonlinear part of generalized-alpha "
        "algorithm\n"
    );

    auto max_iterations = this->time_stepper_.GetMaximumNumberOfIterations();
    for (time_stepper_.SetNumberOfIterations(0);
         time_stepper_.GetNumberOfIterations() < max_iterations;
         time_stepper_.IncrementNumberOfIterations()) {
        log->Debug(
            "Iteration number: " + std::to_string(time_stepper_.GetNumberOfIterations() + 1) + "\n"
        );

        // auto gen_coords_next = gen_coords * quaternion_from_rotation_vector(x);

        auto residuals = ComputeResiduals(gen_coords, vector);
        if (this->CheckConvergence(residuals)) {
            this->is_converged_ = true;
            break;
        }

        auto iteration_matrix = ComputeIterationMatrix(gen_coords, matrix);
        auto delta_x = residuals;
        auto delta_constraints = constraints;

        solve_linear_system(iteration_matrix, delta_x);
        solve_linear_system(iteration_matrix, delta_constraints);

        Kokkos::parallel_for(
            size,
            KOKKOS_LAMBDA(const int i) {
                x(i) += delta_x(i);
                velocity(i) += GAMMA_PRIME * delta_x(i);
                acceleration(i) += BETA_PRIME * delta_x(i);
                constraints(i) += delta_constraints(i);
            }
        );
    }

    auto n_iterations = time_stepper_.GetNumberOfIterations();
    this->time_stepper_.IncrementTotalNumberOfIterations(n_iterations);

    // Update algorithmic acceleration once Newton-Raphson iterations have ended
    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const int i) {
            algo_accleration_next(i) += (1. - kALPHA_F_) / (1. - kALPHA_M_) * acceleration(i);
        }
    );

    log->Debug("Final state after performing Newton-Raphson iterations:\n");
    for (size_t i = 0; i < size; i++) {
        log->Debug(
            std::to_string(gen_coords(i)) + "\t" + std::to_string(velocity(i)) + "\t" +
            std::to_string(acceleration(i)) + "\t" + std::to_string(algo_accleration(i)) + "\n"
        );
    }

    if (this->is_converged_) {
        log->Info(
            "Newton-Raphson iterations converged in " + std::to_string(n_iterations) +
            " iterations\n"
        );
        return {State(gen_coords, velocity, acceleration, algo_accleration), constraints};
    }

    log->Warning(
        "Newton-Raphson iterations failed to converge on a solution after " +
        std::to_string(n_iterations) + " iterations!\n"
    );

    return {State(gen_coords, velocity, acceleration, algo_accleration), constraints};
}

HostView1D GeneralizedAlphaTimeIntegrator::ComputeResiduals(
    HostView1D forces, std::function<HostView1D(size_t)> vector
) {
    // This is a just a placeholder, returns a vector of ones
    // TODO: r^q = M(q) * q'' - f + phi_q^T * lambda
    auto size = forces.extent(0);
    auto residual_vector = vector(size);

    auto log = util::Log::Get();
    log->Debug("Residual vector is " + std::to_string(size) + " x 1 with elements\n");
    for (size_t i = 0; i < size; i++) {
        log->Debug(std::to_string(residual_vector(i)) + "\n");
    }

    return residual_vector;
}

bool GeneralizedAlphaTimeIntegrator::CheckConvergence(HostView1D residual) {
    return false;
}

HostView2D GeneralizedAlphaTimeIntegrator::ComputeIterationMatrix(
    HostView1D gen_coords, std::function<HostView2D(size_t)> matrix
) {
    // This is a just a placeholder, returns an identity matrix for now
    // TODO: S_t = [ (M * beta' + C_t * gamma' + K_t)       Phi_q^T
    //                          Phi_q                         0 ]
    auto size = gen_coords.extent(0);
    auto iteration_matrix = matrix(size);

    auto log = util::Log::Get();
    log->Debug(
        "Iteration matrix is " + std::to_string(size) + " x " + std::to_string(size) +
        " with elements" + "\n"
    );
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            log->Debug(
                "(" + std::to_string(i) + ", " + std::to_string(j) +
                ") : " + std::to_string(iteration_matrix(i, j)) + "\n"
            );
        }
    }

    return iteration_matrix;
}

}  // namespace openturbine::rigid_pendulum
