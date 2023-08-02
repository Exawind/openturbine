#include "src/rigid_pendulum_poc/generalized_alpha_time_integrator.h"

#include "src/rigid_pendulum_poc/heavy_top.h"
#include "src/rigid_pendulum_poc/quaternion.h"
#include "src/rigid_pendulum_poc/solver.h"
#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double alpha_f, double alpha_m, double beta, double gamma, TimeStepper time_stepper,
    ProblemType problem_type, bool precondition
)
    : kALPHA_F_(alpha_f),
      kALPHA_M_(alpha_m),
      kBETA_(beta),
      kGAMMA_(gamma),
      time_stepper_(std::move(time_stepper)),
      problem_type_(problem_type),
      precondition_(precondition) {
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
    size_t n_constraints, std::function<HostView2D(size_t)> iteration_matrix,
    std::function<HostView1D(size_t)> residual_vector
) {
    auto log = util::Log::Get();
    log->Debug(
        "ALPHA_F = " + std::to_string(this->kALPHA_F_) + ", " + "ALPHA_M = " +
        std::to_string(this->kALPHA_M_) + ", " + "BETA = " + std::to_string(this->kBETA_) + ", " +
        "GAMMA = " + std::to_string(this->kGAMMA_) + "\n"
    );

    std::vector<State> states{initial_state};
    auto n_steps = this->time_stepper_.GetNumberOfSteps();
    for (size_t i = 0; i < n_steps; i++) {
        this->time_stepper_.AdvanceTimeStep();
        log->Info("** Integrating step number " + std::to_string(i + 1) + " **\n");
        states.emplace_back(std::get<0>(this->AlphaStep(
            states[i], mass_matrix, gen_forces, n_constraints, iteration_matrix, residual_vector
        )));
    }

    log->Info("Time integration has completed!\n");

    return states;
}

std::tuple<State, HostView1D> GeneralizedAlphaTimeIntegrator::AlphaStep(
    const State& state, const MassMatrix& mass_matrix, const GeneralizedForces& gen_forces,
    size_t n_constraints, std::function<HostView2D(size_t)> matrix,
    std::function<HostView1D(size_t)> vector
) {
    auto gen_coords = state.GetGeneralizedCoordinates();
    auto velocity = state.GetVelocity();
    auto acceleration = state.GetAcceleration();
    auto algo_acceleration = state.GetAlgorithmicAcceleration();

    // Initialize some X_next variables to assist in updating the State - only for ones that
    // require both the current and next values in a calculation
    auto gen_coords_next = HostView1D("gen_coords_next", gen_coords.size());
    auto algo_acceleration_next =
        HostView1D("algorithmic_acceleration_next", algo_acceleration.size());
    auto delta_gen_coords = HostView1D("gen_coords_increment", velocity.size());
    auto lagrange_mults_next = HostView1D("lagrange_mults_next", n_constraints);

    // Perform the linear update part of the generalized alpha algorithm
    const auto h = this->time_stepper_.GetTimeStep();
    const auto size = velocity.size();

    // Algorithm from Table 1, Brüls, Cardona, and Arnold 2012
    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const size_t i) {
            algo_acceleration_next(i) =
                (kALPHA_F_ * acceleration(i) - kALPHA_M_ * algo_acceleration(i)) / (1. - kALPHA_M_);

            delta_gen_coords(i) = velocity(i) + h * (0.5 - kBETA_) * algo_acceleration(i) +
                                  h * kBETA_ * algo_acceleration_next(i);
            velocity(i) +=
                h * (1 - kGAMMA_) * algo_acceleration(i) + h * kGAMMA_ * algo_acceleration_next(i);

            algo_acceleration(i) = algo_acceleration_next(i);

            acceleration(i) = 0.;
        }
    );

    // Initialize lagrange_mults_next to zero separately since it might be of different size
    Kokkos::deep_copy(lagrange_mults_next, 0.);

    auto log = util::Log::Get();
    log->Debug(
        "Initial update of algo_acceleration, algo_acceleration_next, velocity, and "
        "delta_gen_coords:\n"
    );
    for (size_t i = 0; i < size; i++) {
        log->Debug(
            std::to_string(algo_acceleration(i)) + " " + std::to_string(algo_acceleration_next(i)) +
            " " + std::to_string(velocity(i)) + " " + std::to_string(delta_gen_coords(i)) + "\n"
        );
    }

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha algorithm
    log->Info(
        "Performing Newton-Raphson iterations to update solution using the generalized-alpha "
        "algorithm\n"
    );

    const auto BETA_PRIME = (1 - kALPHA_M_) / (h * h * kBETA_ * (1 - kALPHA_F_));
    const auto GAMMA_PRIME = kGAMMA_ / (h * kBETA_);
    log->Debug(
        "BETA_PRIME = " + std::to_string(BETA_PRIME) +
        ", GAMMA_PRIME = " + std::to_string(GAMMA_PRIME) + "\n"
    );

    // Precondition the linear solve (Bottasso et al 2008)
    const auto dl = HostView2D("dl", size + n_constraints, size + n_constraints);
    const auto dr = HostView2D("dr", size + n_constraints, size + n_constraints);
    Kokkos::deep_copy(dl, 0.);
    Kokkos::deep_copy(dr, 0.);

    if (this->precondition_) {
        Kokkos::parallel_for(
            size + n_constraints,
            KOKKOS_LAMBDA(const size_t i) {
                dl(i, i) = 1.;
                dr(i, i) = 1.;
            }
        );

        Kokkos::parallel_for(
            size + n_constraints,
            KOKKOS_LAMBDA(const size_t i) {
                if (i >= size) {
                    dr(i, i) = 1. / (kBETA_ * h * h);
                } else {
                    dl(i, i) = kBETA_ * h * h;
                }
            }
        );
    }

    const auto max_iterations = this->time_stepper_.GetMaximumNumberOfIterations();
    for (time_stepper_.SetNumberOfIterations(0);
         time_stepper_.GetNumberOfIterations() < max_iterations;
         time_stepper_.IncrementNumberOfIterations()) {
        log->Debug(
            "* Iteration number: " + std::to_string(time_stepper_.GetNumberOfIterations() + 1) +
            " *\n"
        );

        gen_coords_next = UpdateGeneralizedCoordinates(gen_coords, delta_gen_coords);

        // Compute the residuals and check for convergence
        const auto residuals = ComputeResiduals(
            mass_matrix, gen_forces, gen_coords_next, velocity, acceleration, lagrange_mults_next,
            vector
        );

        if (this->CheckConvergence(residuals)) {
            this->is_converged_ = true;
            break;
        }

        // Compute the iteration matrix and solve the linear system to get the increments
        auto iteration_matrix = ComputeIterationMatrix(
            BETA_PRIME, GAMMA_PRIME, mass_matrix, gen_forces, gen_coords_next, velocity,
            lagrange_mults_next, h, delta_gen_coords, matrix
        );

        log->Debug("residuals is " + std::to_string(residuals.extent(0)) + " x 1 with elements\n");
        for (size_t i = 0; i < residuals.size(); i++) {
            log->Debug(std::to_string(residuals(i)) + "\n");
        }

        log->Debug(
            "Iteration matrix is " + std::to_string(iteration_matrix.extent(0)) + " x " +
            " with elements" + "\n"
        );
        for (size_t i = 0; i < iteration_matrix.extent(0); i++) {
            for (size_t j = 0; j < iteration_matrix.extent(1); j++) {
                log->Debug(
                    "(" + std::to_string(i) + ", " + std::to_string(j) +
                    ") : " + std::to_string(iteration_matrix(i, j)) + "\n"
                );
            }
        }

        if (this->precondition_) {
            iteration_matrix = multiply_matrix_with_matrix(iteration_matrix, dr);
            iteration_matrix = multiply_matrix_with_matrix(dl, iteration_matrix);

            Kokkos::parallel_for(
                6, KOKKOS_LAMBDA(const size_t i) { residuals(i) = residuals(i) * kBETA_ * h * h; }
            );
        }

        auto soln_increments = HostView1D("soln_increments", residuals.size());
        Kokkos::deep_copy(soln_increments, residuals);
        solve_linear_system(iteration_matrix, soln_increments);

        log->Debug("Solution increments:\n");
        for (size_t i = 0; i < soln_increments.size(); i++) {
            log->Debug(std::to_string(soln_increments(i)) + "\n");
        }

        HostView1D delta_x("delta_x", delta_gen_coords.size());
        Kokkos::parallel_for(
            delta_gen_coords.size(),
            // Take negative of the solution increments to update generalized coordinates
            KOKKOS_LAMBDA(const size_t i) { delta_x(i) = -soln_increments(i); }
        );

        if (n_constraints > 0) {
            HostView1D delta_lagrange_mults("delta_lagrange_mults", n_constraints);

            if (this->precondition_) {
                Kokkos::parallel_for(
                    n_constraints,
                    KOKKOS_LAMBDA(const size_t i) {
                        // Take negative of the solution increments to update Lagrange multipliers
                        delta_lagrange_mults(i) =
                            -soln_increments(i + delta_gen_coords.size()) / (kBETA_ * h * h);
                        lagrange_mults_next(i) += delta_lagrange_mults(i);
                    }
                );
            } else {
                Kokkos::parallel_for(
                    n_constraints,
                    KOKKOS_LAMBDA(const size_t i) {
                        // Take negative of the solution increments to update Lagrange multipliers
                        delta_lagrange_mults(i) = -soln_increments(i + delta_gen_coords.size());
                        lagrange_mults_next(i) += delta_lagrange_mults(i);
                    }
                );
            }
        }

        // Update the velocity, acceleration, and constraints based on the increments
        Kokkos::parallel_for(
            size,
            KOKKOS_LAMBDA(const size_t i) {
                delta_gen_coords(i) += delta_x(i) / h;
                velocity(i) += GAMMA_PRIME * delta_x(i);
                acceleration(i) += BETA_PRIME * delta_x(i);
            }
        );
    }

    const auto n_iterations = time_stepper_.GetNumberOfIterations();
    this->time_stepper_.IncrementTotalNumberOfIterations(n_iterations);

    // Update algorithmic acceleration once Newton-Raphson iterations have ended
    Kokkos::parallel_for(
        size,
        KOKKOS_LAMBDA(const size_t i) {
            algo_acceleration_next(i) += (1. - kALPHA_F_) / (1. - kALPHA_M_) * acceleration(i);
        }
    );

    log->Debug("Final state upon performing Newton-Raphson iterations:\n");
    log->Debug("Generalized coordinates, velocity, acceleration, and algorithmic acceleration \n");
    for (size_t i = 0; i < size; i++) {
        log->Debug(
            std::to_string(gen_coords_next(i)) + ", " + std::to_string(velocity(i)) + ", " +
            std::to_string(acceleration(i)) + ", " + std::to_string(algo_acceleration_next(i)) + "\n"
        );
    }
    log->Debug("Lagrange multipliers:\n");
    for (size_t i = 0; i < n_constraints; i++) {
        log->Debug(std::to_string(lagrange_mults_next(i)) + "\n");
    }

    auto results = std::make_tuple(
        State{gen_coords_next, velocity, acceleration, algo_acceleration_next}, lagrange_mults_next
    );

    if (this->is_converged_) {
        log->Info(
            "Newton-Raphson iterations converged in " + std::to_string(n_iterations + 1) +
            " iterations\n"
        );
        return results;
    }

    log->Warning(
        "Newton-Raphson iterations failed to converge on a solution after " +
        std::to_string(n_iterations + 1) + " iterations!\n"
    );
    return results;
}

HostView1D GeneralizedAlphaTimeIntegrator::UpdateGeneralizedCoordinates(
    const HostView1D gen_coords, const HostView1D delta_gen_coords
) {
    // {gen_coords_next} = {gen_coords} + h * {delta_gen_coords}
    //
    // Step 1: R^3 update, done with vector addition
    auto current_position = Vector{gen_coords(0), gen_coords(1), gen_coords(2)};
    auto updated_position = Vector{delta_gen_coords(0), delta_gen_coords(1), delta_gen_coords(2)};
    const auto h = this->time_stepper_.GetTimeStep();
    auto r = current_position + (updated_position * h);

    auto log = util::Log::Get();
    log->Debug(
        "Updated position: " + std::to_string(r.GetXComponent()) + ", " +
        std::to_string(r.GetYComponent()) + ", " + std::to_string(r.GetZComponent()) + "\n"
    );
    // Step 2: SO(3) update, done with quaternion composition
    Quaternion current_orientation{gen_coords(3), gen_coords(4), gen_coords(5), gen_coords(6)};
    auto update_orientation = quaternion_from_rotation_vector(
        // Convert Vector -> Quaternion via exponential mapping
        Vector{delta_gen_coords(3), delta_gen_coords(4), delta_gen_coords(5)} * h
    );
    auto q = current_orientation * update_orientation;

    log->Debug(
        "Updated orientation: " + std::to_string(q.GetScalarComponent()) + ", " +
        std::to_string(q.GetXComponent()) + ", " + std::to_string(q.GetYComponent()) + ", " +
        std::to_string(q.GetZComponent()) + "\n"
    );

    // Construct the updated generalized coordinates from position and orientation vectors
    auto gen_coords_next = HostView1D("generalized_coordinates_next", gen_coords.size());
    auto components = std::vector<double>{
        r.GetXComponent(),       // component 1
        r.GetYComponent(),       // component 2
        r.GetZComponent(),       // component 3
        q.GetScalarComponent(),  // component 4
        q.GetXComponent(),       // component 5
        q.GetYComponent(),       // component 6
        q.GetZComponent()        // component 7
    };

    Kokkos::parallel_for(
        gen_coords.size(), KOKKOS_LAMBDA(const size_t i) { gen_coords_next(i) = components[i]; }
    );

    return gen_coords_next;
}

HostView1D GeneralizedAlphaTimeIntegrator::ComputeResiduals(
    const MassMatrix& mass, const GeneralizedForces& gen_forces, const HostView1D gen_coords,
    const HostView1D velocity, const HostView1D acceleration, const HostView1D lagrange_mults,
    std::function<HostView1D(size_t)> vector
) {
    // The residual vector consists of two parts
    // {residual} = {
    //     {residual_gen_coords},
    //     {residual_constraints}
    // }
    //
    // {residual_gen_coords} = [M(q)] {v'} + {g(q,v,t)} + [B(q)]T {Lambda}
    // where,
    // [M(q)] = mass matrix
    // {v'} = acceleration vector
    // {g(q,v,t)} = generalized forces vector
    // [B(q)] = constraint gradient matrix
    // {Lambda} = Lagrange multipliers vector
    //
    // {residual_constraints} = {Phi(q)}
    // where,
    // {Phi(q)} = constraint vector

    auto size = acceleration.extent(0) + lagrange_mults.extent(0);
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

            // Generalized forces as defined in Brüls and Cardona 2010
            auto forces = [mass, velocity]() {
                auto m = 15.;
                auto gravity = Vector(0., 0., 9.81);
                auto forces = gravity * m;

                auto angular_velocity = create_vector({
                    velocity(3),  // velocity component 4 -> component 1
                    velocity(4),  // velocity component 5 -> component 2
                    velocity(5)   // velocity component 6 -> component 3
                });
                auto J = mass.GetMomentOfInertiaMatrix();
                auto J_omega = multiply_matrix_with_vector(J, angular_velocity);

                auto angular_velocity_vector =
                    Vector(angular_velocity(0), angular_velocity(1), angular_velocity(2));
                auto J_omega_vector = Vector(J_omega(0), J_omega(1), J_omega(2));
                auto moments = angular_velocity_vector.CrossProduct(J_omega_vector);

                auto generalized_forces = GeneralizedForces(forces, moments);

                return generalized_forces.GetGeneralizedForces();
            };

            auto gen_forces_vector = forces();

            auto position_vector = create_vector(
                // Create vector from appropriate components of generalized coordinates
                {gen_coords(0), gen_coords(1), gen_coords(2)}
            );

            const auto reference_position_vector = create_vector({0., 1., 0});

            residual_vector = heavy_top_residual_vector(
                mass_matrix, rotation_matrix, acceleration, gen_forces_vector, position_vector,
                lagrange_mults, reference_position_vector
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

bool GeneralizedAlphaTimeIntegrator::CheckConvergence(const HostView1D residual) {
    // L2 norm of the residual vector should be very small (< epsilon) for the solution
    // to be considered converged
    double residual_norm = 0.;
    Kokkos::parallel_reduce(
        residual.extent(0),
        KOKKOS_LAMBDA(int i, double& residual_partial_sum) {
            double residual_value = residual(i);
            residual_partial_sum += residual_value * residual_value;
        },
        Kokkos::Sum<double>(residual_norm)
    );
    residual_norm = std::sqrt(residual_norm);

    auto log = util::Log::Get();
    log->Debug("Residual norm: " + std::to_string(residual_norm) + "\n");

    return (residual_norm) < kCONVERGENCETOLERANCE ? true : false;
}

HostView2D GeneralizedAlphaTimeIntegrator::ComputeIterationMatrix(
    const double& BETA_PRIME, const double& GAMMA_PRIME, const MassMatrix& mass,
    const GeneralizedForces& gen_forces, const HostView1D gen_coords, const HostView1D velocity,
    const HostView1D lagrange_mults, const double& h, const HostView1D delta_gen_coords,
    std::function<HostView2D(size_t)> matrix
) {
    // Iteration matrix for the generalized alpha algorithm is given by
    // [iteration matrix] = [
    //     [M(q)] * beta' + [C_t(q,v,t)] * gamma' + [K_t(q,v,v',Lambda,t)]    [B(q)^T]]
    //                            [ B(q) ]                                       [0]
    // ]
    // where,
    // [M(q)] = mass matrix
    // [C_t(q,v,t)] = Tangent damping matrix
    // [K_t(q,v,v',Lambda,t)] = Tangent stiffness matrix
    // [B(q)] = Constraint gradeint matrix

    auto size = velocity.extent(0) + lagrange_mults.extent(0);
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

            const HostView1D reference_position_vector = create_vector({0., 1., 0});

            iteration_matrix = heavy_top_iteration_matrix(
                BETA_PRIME, GAMMA_PRIME, mass_matrix, inertia_matrix, rotation_matrix,
                angular_velocity_vector, lagrange_mults, reference_position_vector, h,
                delta_gen_coords
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
