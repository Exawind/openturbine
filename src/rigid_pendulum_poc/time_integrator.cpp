#include "src/rigid_pendulum_poc/time_integrator.h"

#include "src/rigid_pendulum_poc/solver.h"
#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

HostView2D create_identity_matrix(size_t size) {
    auto matrix = HostView2D("matrix", size, size);
    auto diagonal_entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size);
    auto fill_diagonal = [matrix](int index) {
        matrix(index, index) = 1.;
    };

    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

    return matrix;
}

HostView1D create_identity_vector(size_t size) {
    auto vector = HostView1D("vector", size);

    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(int i) { vector(i) = 1.; }
    );

    return vector;
}

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double initial_time, double time_step, size_t n_steps, double alpha_f, double alpha_m,
    double beta, double gamma, size_t max_iterations
)
    : initial_time_(initial_time),
      time_step_(time_step),
      n_steps_(n_steps),
      kALPHA_F_(alpha_f),
      kALPHA_M_(alpha_m),
      kBETA_(beta),
      kGAMMA_(gamma),
      kMAX_ITERATIONS_(max_iterations) {
    this->current_time_ = initial_time;
    this->n_iterations_ = 0;
    this->total_n_iterations_ = 0;

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

    if (this->kMAX_ITERATIONS_ < 1) {
        throw std::invalid_argument("Invalid value for max_iterations");
    }

    this->is_converged_ = false;
}

std::vector<State> GeneralizedAlphaTimeIntegrator::Integrate(
    const State& initial_state, std::function<HostView2D(size_t)> matrix,
    std::function<HostView1D(size_t)> vector
) {
    auto log = util::Log::Get();

    std::vector<State> states{initial_state};
    for (size_t i = 0; i < this->n_steps_; i++) {
        log->Info("Integrating step number " + std::to_string(i + 1) + "\n");
        states.emplace_back(std::get<0>(this->AlphaStep(states[i], matrix, vector)));
        this->AdvanceTimeStep();
    }

    log->Info("Time integration completed successfully!\n");

    return states;
}

std::tuple<State, HostView1D> GeneralizedAlphaTimeIntegrator::AlphaStep(
    const State& state, std::function<HostView2D(size_t)> matrix,
    std::function<HostView1D(size_t)> vector
) {
    auto linear_update = UpdateLinearSolution(state);
    auto gen_coords = linear_update.GetGeneralizedCoordinates();
    auto gen_velocity = linear_update.GetGeneralizedVelocity();
    auto gen_accln = linear_update.GetGeneralizedAcceleration();
    auto algo_accln = linear_update.GetAlgorithmicAcceleration();

    const auto h = this->time_step_;
    const auto size = gen_coords.size();
    const double beta_prime = (1 - kALPHA_M_) / (h * h * kBETA_ * (1 - kALPHA_F_));
    const double gamma_prime = kGAMMA_ / (h * kBETA_);

    // TODO: Provide actual constraints
    auto constraints = HostView1D("constraints", size);

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha algorithm
    auto log = util::Log::Get();
    log->Info(
        "Performing Newton-Raphson iterations to update the nonlinear part of generalized-alpha "
        "algorithm\n"
    );

    for (n_iterations_ = 0; n_iterations_ < this->kMAX_ITERATIONS_; n_iterations_++) {
        log->Debug("Iteration number: " + std::to_string(this->n_iterations_ + 1) + "\n");

        auto residuals = ComputeResiduals(gen_coords, vector);
        // TODO: Provide actual force increments
        auto increments = residuals;

        if (this->CheckConvergence(residuals, increments)) {
            this->is_converged_ = true;
            break;
        }

        auto iteration_matrix = ComputeIterationMatrix(gen_coords, matrix);
        auto gen_coords_delta = residuals;
        auto constraints_delta = constraints;
        solve_linear_system(iteration_matrix, gen_coords_delta);
        solve_linear_system(iteration_matrix, constraints_delta);

        Kokkos::parallel_for(
            size,
            KOKKOS_LAMBDA(const int i) {
                gen_coords(i) += gen_coords_delta(i);
                gen_velocity(i) += gamma_prime * gen_coords_delta(i);
                gen_accln(i) += beta_prime * gen_coords_delta(i);
                constraints(i) += constraints_delta(i);
            }
        );
    }

    this->total_n_iterations_ += this->n_iterations_;

    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const int i) {
            // Update algorithmic acceleration once soln has converged
            algo_accln(i) += (1 - kALPHA_F_) / (1 - kALPHA_M_) * gen_accln(i);
        }
    );

    if (this->is_converged_) {
        log->Info(
            "Newton-Raphson iterations converged in " + std::to_string(this->n_iterations_) +
            " iterations\n"
        );
    } else {
        log->Warning(
            "Newton-Raphson iterations failed to converge on a solution after " +
            std::to_string(this->n_iterations_) + " iterations!\n"
        );
    }

    log->Debug("Final state after performing Newton-Raphson iterations:\n");
    for (size_t i = 0; i < size; i++) {
        log->Debug(
            std::to_string(gen_coords(i)) + "\t" + std::to_string(gen_velocity(i)) + "\t" +
            std::to_string(gen_accln(i)) + "\t" + std::to_string(algo_accln(i)) + "\n"
        );
    }

    return {State(gen_coords, gen_velocity, gen_accln, algo_accln), constraints};
}

State GeneralizedAlphaTimeIntegrator::UpdateLinearSolution(const State& state) {
    auto gen_coords = state.GetGeneralizedCoordinates();
    auto gen_velocity = state.GetGeneralizedVelocity();
    auto gen_accln = state.GetGeneralizedAcceleration();
    auto algo_accln = state.GetAlgorithmicAcceleration();

    // Update generalized coordinates, generalized velocity, and algorithmic acceleration
    // based on generalized coordinates, generalized velocity, generalized acceleration,
    // and algorithmic acceleration from previous time step and algorithmic acceleration
    // from current time step
    const auto h = this->time_step_;
    const auto size = gen_coords.size();

    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const int i) {
            gen_coords(i) += h * gen_velocity(i) + h * h * (0.5 - kBETA_) * algo_accln(i);
            gen_velocity(i) += h * (1 - kGAMMA_) * algo_accln(i);
            algo_accln(i) =
                (1.0 / (1.0 - kALPHA_M_)) * (kALPHA_F_ * gen_accln(i) - kALPHA_M_ * algo_accln(i));
            gen_coords(i) += h * h * kBETA_ * algo_accln(i);
            gen_velocity(i) += h * kBETA_ * algo_accln(i);
            gen_accln(i) = 0.;
        }
    );

    auto log = util::Log::Get();
    log->Debug("Linear solution: gen_coords, gen_velocity, algo_acceleration\n");
    for (size_t i = 0; i < size; i++) {
        log->Debug(
            "row " + std::to_string(i) + ": " + std::to_string(gen_coords(i)) + "\t" +
            std::to_string(gen_velocity(i)) + "\t" + std::to_string(algo_accln(i)) + "\n"
        );
    }

    return State(gen_coords, gen_velocity, gen_accln, algo_accln);
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

bool GeneralizedAlphaTimeIntegrator::CheckConvergence(HostView1D residual, HostView1D increment) {
    // L2 norm of the residual load vector should be very small (< epsilon) compared to the
    // L2 norm of the load vector increment for the solution to be converged
    double residual_norm = 0.;
    double increment_norm = 0.;

    Kokkos::parallel_reduce(
        residual.extent(0),
        KOKKOS_LAMBDA(int i, double& residual_partial_sum, double& increment_partial_sum) {
            double residual_value = residual(i);
            double increment_value = increment(i);

            residual_partial_sum += residual_value * residual_value;
            increment_partial_sum += increment_value * increment_value;
        },
        Kokkos::Sum<double>(residual_norm), Kokkos::Sum<double>(increment_norm)
    );

    residual_norm = std::sqrt(residual_norm);
    increment_norm = std::sqrt(increment_norm);

    auto log = util::Log::Get();
    log->Debug(
        "Residual norm: " + std::to_string(residual_norm) + ", " +
        "Increment norm: " + std::to_string(increment_norm) + "\n"
    );

    return (residual_norm / increment_norm) < kTOLERANCE ? true : false;
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
