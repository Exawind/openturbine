#include "src/rigid_pendulum_poc/generalized_alpha_time_integrator.h"

#include "src/rigid_pendulum_poc/heavy_top.h"
#include "src/rigid_pendulum_poc/quaternion.h"
#include "src/rigid_pendulum_poc/solver.h"
#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double alpha_f, double alpha_m, double beta, double gamma, TimeStepper time_stepper,
    ProblemType problem_type
)
    : kALPHA_F_(alpha_f),
      kALPHA_M_(alpha_m),
      kBETA_(beta),
      kGAMMA_(gamma),
      time_stepper_(std::move(time_stepper)),
      problem_type_(problem_type) {
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
    const State& initial_state, const MassMatrix& mass_matrix, const GeneralizedForces& gen_forces,
    HostView1D lagrange_multipliers, std::function<HostView2D(size_t)> iteration_matrix,
    std::function<HostView1D(size_t)> residual_vector
) {
    auto log = util::Log::Get();

    std::vector<State> states{initial_state};
    auto n_steps = this->time_stepper_.GetNumberOfSteps();
    for (size_t i = 0; i < n_steps; i++) {
        log->Info("Integrating step number " + std::to_string(i + 1) + "\n");
        states.emplace_back(std::get<0>(this->AlphaStep(
            states[i], mass_matrix, gen_forces, lagrange_multipliers, iteration_matrix,
            residual_vector
        )));
        this->time_stepper_.AdvanceTimeStep();
    }

    log->Info("Time integration completed successfully!\n");

    return states;
}

std::tuple<State, HostView1D> GeneralizedAlphaTimeIntegrator::AlphaStep(
    const State& state, const MassMatrix& mass_matrix, const GeneralizedForces& gen_forces,
    HostView1D lagrange_multipliers, std::function<HostView2D(size_t)> matrix,
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

    // Perform the linear update part of the generalized alpha algorithm
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

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha algorithm
    auto log = util::Log::Get();
    log->Info(
        "Performing Newton-Raphson iterations to update the nonlinear part of generalized-alpha "
        "algorithm\n"
    );

    const auto BETA_PRIME = (1 - kALPHA_M_) / (h * h * kBETA_ * (1 - kALPHA_F_));
    const auto GAMMA_PRIME = kGAMMA_ / (h * kBETA_);

    auto max_iterations = this->time_stepper_.GetMaximumNumberOfIterations();
    for (time_stepper_.SetNumberOfIterations(0);
         time_stepper_.GetNumberOfIterations() < max_iterations;
         time_stepper_.IncrementNumberOfIterations()) {
        log->Debug(
            "Iteration number: " + std::to_string(time_stepper_.GetNumberOfIterations() + 1) + "\n"
        );

        auto gen_coords_next = ComputeUpdatedGeneralizedCoordinates(gen_coords, x);

        // Compute the residuals and check for convergence
        auto residuals = ComputeResiduals(
            mass_matrix, gen_forces, gen_coords_next, acceleration, lagrange_multipliers, vector
        );
        if (this->CheckConvergence(residuals)) {
            this->is_converged_ = true;
            break;
        }

        // Compute the iteration matrix and solve the linear system to get the increments
        auto iteration_matrix = ComputeIterationMatrix(
            BETA_PRIME, GAMMA_PRIME, mass_matrix, gen_forces, gen_coords_next, velocity,
            acceleration, lagrange_multipliers, matrix
        );
        auto delta_x = residuals;
        auto delta_constraints = constraints;

        solve_linear_system(iteration_matrix, delta_x);
        solve_linear_system(iteration_matrix, delta_constraints);

        // Update the velocity, acceleration, and constraints based on the increments
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

HostView1D GeneralizedAlphaTimeIntegrator::ComputeUpdatedGeneralizedCoordinates(
    HostView1D gen_coords, HostView1D x
) {
    // Construct the updated generalized coordinates from position and orientation vectors
    auto current_position = Vector{gen_coords(0), gen_coords(1), gen_coords(2)};
    auto updated_position = Vector{x(0), x(1), x(2)};
    auto r = current_position + updated_position;

    Quaternion current_orientation{gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6)};
    auto update_orientation = quaternion_from_rotation_vector(Vector{x(3), x(4), x(5)});
    auto q = current_orientation * update_orientation;

    auto gen_coords_next = HostView1D("generalized_coordinates_next", gen_coords.size());
    gen_coords_next(0) = r.GetXComponent();
    gen_coords_next(1) = r.GetYComponent();
    gen_coords_next(2) = r.GetZComponent();
    gen_coords_next(3) = q.GetScalarComponent();
    gen_coords_next(4) = q.GetXComponent();
    gen_coords_next(5) = q.GetYComponent();
    gen_coords_next(6) = q.GetZComponent();

    return gen_coords_next;
}

HostView1D GeneralizedAlphaTimeIntegrator::ComputeResiduals(
    const MassMatrix& mass, const GeneralizedForces& gen_forces, HostView1D gen_coords,
    HostView1D acceleration, HostView1D lagrange_multipliers,
    std::function<HostView1D(size_t)> vector
) {
    auto size = acceleration.extent(0);
    auto residual_vector = vector(size);

    switch (this->problem_type_) {
        // Heavy top problem
        case ProblemType::kHeavyTop: {
            auto mass_matrix = mass.GetMassMatrix();

            // Convert the quaternion representing orientation -> rotation matrix
            auto RM = quaternion_to_rotation_matrix(
                // Create quaternion from appropriate components of generalized coordinates
                Quaternion{gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6)}
            );
            auto [m00, m01, m02] = std::get<0>(RM).GetComponents();
            auto [m10, m11, m12] = std::get<1>(RM).GetComponents();
            auto [m20, m21, m22] = std::get<2>(RM).GetComponents();
            auto rotation_matrix =
                create_matrix({{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}});

            auto gen_forces_vector = gen_forces.GetGeneralizedForces();
            auto position_vector = create_vector({gen_coords(0), gen_coords(1), gen_coords(2)});

            residual_vector = heavy_top_residual_vector(
                mass_matrix, rotation_matrix, acceleration, gen_forces_vector, position_vector,
                lagrange_multipliers
            );
        } break;
        // Rigid pendulum problem
        case ProblemType::kRigidPendulum: {
            residual_vector = rigid_pendulum_residual_vector(size);
        } break;
        // Rigid body problem
        case ProblemType::kRigidBody: {
            // Do nothing, we're using it for unit testing at the moment
        } break;
        // Default case
        default:
            throw std::runtime_error("Invalid problem type!");
    }

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
    const double& BETA_PRIME, const double& GAMMA_PRIME, const MassMatrix& mass,
    const GeneralizedForces& gen_forces, HostView1D gen_coords, HostView1D velocity,
    HostView1D acceleration, HostView1D lagrange_multipliers,
    std::function<HostView2D(size_t)> matrix
) {
    auto size = gen_coords.extent(0);
    auto iteration_matrix = matrix(size);

    switch (this->problem_type_) {
        // Heavy top problem
        case ProblemType::kHeavyTop: {
            auto mass_matrix = mass.GetMassMatrix();
            auto inertia_matrix = mass.GetMomentOfInertiaMatrix();

            // Convert the quaternion representing orientation -> rotation matrix
            auto RM = quaternion_to_rotation_matrix(
                // Create quaternion from appropriate components of generalized coordinates
                Quaternion{gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6)}
            );
            auto [m00, m01, m02] = std::get<0>(RM).GetComponents();
            auto [m10, m11, m12] = std::get<1>(RM).GetComponents();
            auto [m20, m21, m22] = std::get<2>(RM).GetComponents();
            auto rotation_matrix =
                create_matrix({{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}});

            auto gen_forces_vector = gen_forces.GetGeneralizedForces();
            auto angular_velocity_vector = create_vector({velocity(3), velocity(4), velocity(5)});
            auto position_vector = create_vector({gen_coords(0), gen_coords(1), gen_coords(2)});

            iteration_matrix = heavy_top_iteration_matrix(
                BETA_PRIME, GAMMA_PRIME, mass_matrix, inertia_matrix, rotation_matrix,
                angular_velocity_vector, position_vector, lagrange_multipliers
            );
        } break;
        // Rigid pendulum problem
        case ProblemType::kRigidPendulum: {
            iteration_matrix = rigid_pendulum_iteration_matrix(size);
        } break;
        // Rigid body problem
        case ProblemType::kRigidBody: {
            // Do nothing, we're using it for unit testing at the moment
        } break;
        // Default case
        default:
            throw std::runtime_error("Problem type not supported!");
    }

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
