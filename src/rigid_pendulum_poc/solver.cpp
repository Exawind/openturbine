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
    log->Debug(
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

    log->Debug("LAPACKE_dgesv returned exit code " + std::to_string(info) + "\n");

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
    double initial_time, double time_step, size_t number_of_steps, bool nonlinear_analysis
)
    : initial_time_(initial_time),
      time_step_(time_step),
      number_of_steps_(number_of_steps),
      nonlinear_analysis_(nonlinear_analysis) {
    this->current_time_ = initial_time;
    this->n_iterations_ = 0;
    this->total_n_iterations_ = 0;
}

std::vector<State> GeneralizedAlphaTimeIntegrator::Integrate(const State& initial_state) {
    auto log = util::Log::Get();

    std::vector<State> states{initial_state};

    for (size_t i = 0; i < this->number_of_steps_; i++) {
        log->Debug("Integrating step " + std::to_string(i + 1) + "\n");
        states.emplace_back(this->AlphaStep(states[i]));
        this->AdvanceTimeStep();
    }

    log->Debug("Time integration completed successfully!\n");

    return states;
}

State GeneralizedAlphaTimeIntegrator::AlphaStep(const State& state) {
    auto log = util::Log::Get();
    log->Debug("Attempting the AlphaStep...\n");

    auto gen_coords = state.GetGeneralizedCoordinates();
    auto gen_velocity = state.GetGeneralizedVelocity();
    auto gen_accln = state.GetGeneralizedAcceleration();
    auto algo_accln = state.GetAlgorithmicAcceleration();

    auto [linear_coords, linear_velocity, linear_algo_accln] =
        UpdateLinearSolution(gen_coords, gen_velocity, gen_accln, algo_accln);

    gen_coords = linear_coords;
    gen_velocity = linear_velocity;
    algo_accln = linear_algo_accln;

    const auto h = this->time_step_;
    const auto size = gen_coords.size();
    const double beta_prime = (1 - kALPHA_M) / (h * h * kBETA * (1 - kALPHA_F));
    const double gamma_prime = kGAMMA / (h * kBETA);

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha algorithm
    for (n_iterations_ = 0; n_iterations_ < kMAX_ITERATIONS; n_iterations_++) {
        log->Debug("Iteration: " + std::to_string(this->n_iterations_ + 1) + "\n");

        auto residuals = ComputeResiduals(gen_coords);
        // TODO: Provide actual force increments
        auto increments = residuals;

        if (this->CheckConvergence(residuals, increments)) {
            break;
        }

        auto iteration_matrix = ComputeIterationMatrix(gen_coords);
        auto gen_coords_delta = residuals;
        solve_linear_system(iteration_matrix, gen_coords_delta);

        Kokkos::parallel_for(
            size,
            KOKKOS_LAMBDA(const int i) {
                gen_coords(i) += gen_coords_delta(i);
                gen_velocity(i) += gamma_prime * gen_coords_delta(i);
                gen_accln(i) += beta_prime * gen_coords_delta(i);
            }
        );
    }

    log->Debug("Converged in " + std::to_string(this->n_iterations_) + " iterations\n");
    this->total_n_iterations_ += this->n_iterations_;

    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const int i) {
            // update algorithmic acceleration
            algo_accln(i) += (1 - kALPHA_F) / (1 - kALPHA_M) * gen_accln(i);
        }
    );

    return State(gen_coords, gen_velocity, gen_accln, algo_accln);
}

std::tuple<HostView1D, HostView1D, HostView1D> GeneralizedAlphaTimeIntegrator::UpdateLinearSolution(
    HostView1D gen_coords, HostView1D gen_velocity, HostView1D gen_accln, HostView1D algo_accln
) {
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

    return {gen_coords, gen_velocity, algo_accln};
}

HostView1D GeneralizedAlphaTimeIntegrator::ComputeResiduals(HostView1D forces) {
    // TODO: Compute the residuals
    // r^q = M(q) * q'' - f + phi_q^T * lambda
    // This is a just a placeholder, returns the provided forces as the residuals
    return forces;
}

bool GeneralizedAlphaTimeIntegrator::CheckConvergence(HostView1D residual, HostView1D increment) {
    // L2 norm of the residual load vector should be very small (< epsilon) compared to the
    // L2 norm of the load vector increment for the solution to be converged
    double residual_norm = 0.0;
    double increment_norm = 0.0;

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

    return (residual_norm / increment_norm) < kTOLERANCE ? true : false;
}

HostView2D GeneralizedAlphaTimeIntegrator::ComputeIterationMatrix(HostView1D gen_coords) {
    // TODO: S_t = [ (M * beta' + C_t * gamma' + K_t)       Phi_q^T
    //                          Phi_q                         0 ]
    // This is a just a placeholder, returns an identity matrix for now
    auto matrix = HostView2D("matrix", gen_coords.size(), gen_coords.size());
    auto diagonal_entries =
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, gen_coords.size());
    auto fill_diagonal = [matrix](int index) {
        matrix(index, index) = 1.;
    };

    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

    return matrix;
}

}  // namespace openturbine::rigid_pendulum
