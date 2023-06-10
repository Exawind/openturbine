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
        LAPACK_ROW_MAJOR,           // input: matrix_layout
        rows,                       // input: number of linear equations
        right_hand_sides,           // input: number of rhs
        system.data(),              // input/output: Upon entry, the nxn coefficient matrix
                                    // Upon exit, the factors L and U from the factorization
        leading_dimension_sytem,    // input: leading dimension of system
        pivots.data(),              // output: pivot indices
        solution.data(),            // input/output: Upon entry, the right-hand side matrix
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
      generalized_accelerations_("generalized_accelerations", 1),
      algorithmic_accelerations_("algorithmic_accelerations", 1) {
}

State::State(
    HostView1D gen_coords, HostView1D gen_velocity, HostView1D gen_accln, HostView1D algo_accln
)
    : generalized_coords_("generalized_coordinates", gen_coords.size()),
      generalized_velocity_("generalized_velocity", gen_velocity.size()),
      generalized_accelerations_("generalized_accelerations", gen_accln.size()),
      algorithmic_accelerations_("algorithmic_accelerations", algo_accln.size()) {
    Kokkos::deep_copy(generalized_coords_, gen_coords);
    Kokkos::deep_copy(generalized_velocity_, gen_velocity);
    Kokkos::deep_copy(generalized_accelerations_, gen_accln);
    Kokkos::deep_copy(algorithmic_accelerations_, algo_accln);
}

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double initial_time, double time_step, size_t number_of_steps, State initial_state,
    State state_increment
)
    : initial_time_(initial_time),
      time_step_(time_step),
      number_of_steps_(number_of_steps),
      state_(initial_state),
      state_increment_(state_increment) {
    this->current_time_ = initial_time;
}

void GeneralizedAlphaTimeIntegrator::Integrate() {
    auto log = util::Log::Get();

    // Perform the time integration sequentially
    for (size_t i = 0; i < this->number_of_steps_; i++) {
        log->Debug("Integrating step " + std::to_string(i + 1) + "\n");

        // Perform the alpha step and update the current state
        this->AlphaStep();
        // this->state_ += this->state_increment_;

        // Advance the time of the analysis
        this->AdvanceTimeStep();
    }

    log->Debug("Time integration completed successfully \n");
}

void GeneralizedAlphaTimeIntegrator::UpdateLinearSolution() {
    auto gen_coords = this->state_.GetGeneralizedCoordinates();
    auto gen_velocity = this->state_.GetGeneralizedVelocity();
    auto gen_accln = this->state_.GetGeneralizedAcceleration();
    auto algo_accln = this->state_.GetAccelerations();
    auto h = this->time_step_;

    auto size = gen_coords.size();
    HostView1D gen_coords_next("gen_coords_next", size);
    HostView1D gen_velocity_next("gen_velocity_next", size);
    HostView1D algo_accln_next("algo_accln_next", size);

    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const int i) {
            gen_coords_next(i) =
                gen_coords(i) + h * gen_velocity(i) + h * h * (0.5 - kBETA) * algo_accln(i);
            gen_velocity_next(i) = gen_velocity(i) + h * (1 - kGAMMA) * algo_accln(i);
            algo_accln_next(i) =
                (1.0 / (1.0 - kALPHA_M)) * (kALPHA_F * gen_accln(i) - kALPHA_M * algo_accln(i));
            gen_coords_next(i) = gen_coords_next(i) + h * h * kBETA * algo_accln_next(i);
            gen_velocity_next(i) = gen_velocity_next(i) + h * kBETA * algo_accln_next(i);
        }
    );
}

void GeneralizedAlphaTimeIntegrator::AlphaStep() {
    // Perform the linear update
    UpdateLinearSolution();

    // TODO: Perform the nonlinear update
}

}  // namespace openturbine::rigid_pendulum
