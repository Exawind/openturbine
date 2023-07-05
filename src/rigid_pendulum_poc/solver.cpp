#include "src/rigid_pendulum_poc/solver.h"

#include <lapacke.h>

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

void solve_linear_system(HostView2D system, HostView1D solution) {
    auto rows = static_cast<int>(system.extent(0));
    auto columns = static_cast<int>(system.extent(1));

    if (rows != columns) {
        throw std::invalid_argument("Provided system must be a square matrix");
    }

    if (rows != static_cast<int>(solution.extent(0))) {
        throw std::invalid_argument(
            "Provided system and solution must contain the same number of rows"
        );
    }

    int right_hand_sides{1};
    int leading_dimension_sytem{rows};
    auto pivots = Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace>("pivots", solution.size());
    int leading_dimension_solution{1};

    auto log = util::Log::Get();
    log->Info(
        "Solving a " + std::to_string(rows) + " x " + std::to_string(rows) +
        " system of linear equations with LAPACKE_dgesv" + "\n"
    );

    // Call DGESV from LAPACK to compute the solution to a real system of linear
    // equations A * x = b, returns 0 if successful
    // https://www.netlib.org/lapack/lapacke.html
    auto info = LAPACKE_dgesv(
        LAPACK_ROW_MAJOR,           // input: matrix layout
        rows,                       // input: number of linear equations
        right_hand_sides,           // input: number of rhs
        system.data(),              // input/output: Upon entry, the n x n coefficient matrix
                                    // Upon exit, the factors L and U from the factorization
        leading_dimension_sytem,    // input: leading dimension of system
        pivots.data(),              // output: pivot indices
        solution.data(),            // input/output: Upon entry, the right hand side matrix
                                    // Upon exit, the solution matrix
        leading_dimension_solution  // input: leading dimension of solution
    );

    log->Info("LAPACKE_dgesv returned exit code " + std::to_string(info) + "\n");

    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgesv failed to solve the system!");
    }
}

State::State()
    : generalized_coords_("generalized_coordinates", 1),
      generalized_velocity_("generalized_velocity", 1),
      generalized_acceleration_("generalized_accelerations", 1),
      algorithmic_acceleration_("algorithmic_accelerations", 1) {
}

State::State(
    HostView1D gen_coords, HostView1D gen_velocity, HostView1D gen_accln, HostView1D algo_accln
)
    : generalized_coords_("generalized_coordinates", gen_coords.size()),
      generalized_velocity_("generalized_velocity", gen_velocity.size()),
      generalized_acceleration_("generalized_accelerations", gen_accln.size()),
      algorithmic_acceleration_("algorithmic_accelerations", algo_accln.size()) {
    Kokkos::deep_copy(generalized_coords_, gen_coords);
    Kokkos::deep_copy(generalized_velocity_, gen_velocity);
    Kokkos::deep_copy(generalized_acceleration_, gen_accln);
    Kokkos::deep_copy(algorithmic_acceleration_, algo_accln);
}

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double initial_time, double time_step, size_t n_steps, double alpha_f, double alpha_m,
    double beta, double gamma, size_t max_iterations
)
    : initial_time_(initial_time),
      time_step_(time_step),
      n_steps_(n_steps),
      kALPHA_F(alpha_f),
      kALPHA_M(alpha_m),
      kBETA(beta),
      kGAMMA(gamma),
      kMAX_ITERATIONS(max_iterations) {
    this->current_time_ = initial_time;
    this->n_iterations_ = 0;
    this->total_n_iterations_ = 0;

    if (this->kALPHA_F < 0 || this->kALPHA_F > 1) {
        throw std::invalid_argument("Invalid value for alpha_f");
    }

    if (this->kALPHA_M < 0 || this->kALPHA_M > 1) {
        throw std::invalid_argument("Invalid value for alpha_m");
    }

    if (this->kBETA < 0 || this->kBETA > 0.50) {
        throw std::invalid_argument("Invalid value for beta");
    }

    if (this->kGAMMA < 0 || this->kGAMMA > 1) {
        throw std::invalid_argument("Invalid value for gamma");
    }

    if (this->kMAX_ITERATIONS < 1) {
        throw std::invalid_argument("Invalid value for max_iterations");
    }
}

std::vector<State> GeneralizedAlphaTimeIntegrator::Integrate(const State& initial_state) {
    auto log = util::Log::Get();

    std::vector<State> states{initial_state};
    for (size_t i = 0; i < this->n_steps_; i++) {
        log->Info("Integrating step number " + std::to_string(i + 1) + "\n");
        states.emplace_back(std::get<0>(this->AlphaStep(states[i])));
        this->AdvanceTimeStep();
    }

    log->Info("Time integration completed successfully!\n");

    return states;
}

std::tuple<State, HostView1D> GeneralizedAlphaTimeIntegrator::AlphaStep(const State& state) {
    auto linear_update = UpdateLinearSolution(state);
    auto gen_coords = linear_update.GetGeneralizedCoordinates();
    auto gen_velocity = linear_update.GetGeneralizedVelocity();
    auto gen_accln = linear_update.GetGeneralizedAcceleration();
    auto algo_accln = linear_update.GetAlgorithmicAcceleration();

    const auto h = this->time_step_;
    const auto size = gen_coords.size();
    const double beta_prime = (1 - kALPHA_M) / (h * h * kBETA * (1 - kALPHA_F));
    const double gamma_prime = kGAMMA / (h * kBETA);

    // TODO: Provide actual constraints
    auto constraints = HostView1D("constraints", size);

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha algorithm
    auto log = util::Log::Get();
    log->Info(
        "Performing Newton-Raphson iterations to update the nonlinear part of generalized-alpha "
        "algorithm\n"
    );

    for (n_iterations_ = 0; n_iterations_ < kMAX_ITERATIONS; n_iterations_++) {
        log->Debug("Iteration number: " + std::to_string(this->n_iterations_ + 1) + "\n");

        auto residuals = ComputeResiduals(gen_coords);
        // TODO: Provide actual force increments
        auto increments = residuals;

        if (this->CheckConvergence(residuals, increments)) {
            break;
        }

        auto iteration_matrix = ComputeIterationMatrix(gen_coords);
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

    if (this->n_iterations_ < kMAX_ITERATIONS) {
        log->Info("Converged in " + std::to_string(this->n_iterations_) + " iterations\n");
    }

    if (this->n_iterations_ == kMAX_ITERATIONS) {
        log->Warning("Newton-Raphson iterations failed to converge on a solution!\n");
    }

    this->total_n_iterations_ += this->n_iterations_;

    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const int i) {
            // Update algorithmic acceleration once soln has converged
            algo_accln(i) += (1 - kALPHA_F) / (1 - kALPHA_M) * gen_accln(i);
        }
    );

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
            gen_coords(i) += h * gen_velocity(i) + h * h * (0.5 - kBETA) * algo_accln(i);
            gen_velocity(i) += h * (1 - kGAMMA) * algo_accln(i);
            algo_accln(i) =
                (1.0 / (1.0 - kALPHA_M)) * (kALPHA_F * gen_accln(i) - kALPHA_M * algo_accln(i));
            gen_coords(i) += h * h * kBETA * algo_accln(i);
            gen_velocity(i) += h * kBETA * algo_accln(i);
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

HostView1D GeneralizedAlphaTimeIntegrator::ComputeResiduals(HostView1D forces) {
    // This is a just a placeholder, returns a vector of ones
    // TODO: r^q = M(q) * q'' - f + phi_q^T * lambda
    auto size = forces.extent(0);
    auto residual_vector = HostView1D("residual_vector", size);
    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(int i) { residual_vector(i) = 1.; }
    );

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

HostView2D GeneralizedAlphaTimeIntegrator::ComputeIterationMatrix(HostView1D gen_coords) {
    // This is a just a placeholder, returns an identity matrix for now
    // TODO: S_t = [ (M * beta' + C_t * gamma' + K_t)       Phi_q^T
    //                          Phi_q                         0 ]
    auto size = gen_coords.extent(0);
    auto iteration_matrix = HostView2D("iteration_matrix", size, size);
    auto diagonal_entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size);
    auto fill_diagonal = [iteration_matrix](int index) {
        iteration_matrix(index, index) = 1.;
    };
    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

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
