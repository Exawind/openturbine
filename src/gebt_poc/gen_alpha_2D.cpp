#include "src/gebt_poc/gen_alpha_2D.h"

#include <KokkosBlas.hpp>

#include "src/gebt_poc/linear_solver.h"
#include "src/gen_alpha_poc/heavy_top.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/vector.h"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double alpha_f, double alpha_m, double beta, double gamma,
    gen_alpha_solver::TimeStepper time_stepper, bool precondition, ProblemType problem_type
)
    : kAlphaF_(alpha_f),
      kAlphaM_(alpha_m),
      kBeta_(beta),
      kGamma_(gamma),
      time_stepper_(std::move(time_stepper)),
      is_preconditioned_(precondition),
      problem_type_(problem_type) {
    this->is_converged_ = false;
    this->reference_energy_ = 0.;
}

std::vector<State> GeneralizedAlphaTimeIntegrator::Integrate(
    const State& initial_state, size_t n_constraints,
    std::shared_ptr<LinearizationParameters> linearization_parameters
) {
    auto log = util::Log::Get();
    std::vector<State> states{initial_state};
    auto n_steps = this->time_stepper_.GetNumberOfSteps();
    for (size_t i = 0; i < n_steps; i++) {
        this->time_stepper_.AdvanceTimeStep();
        auto input_state = State{
            states[i].GetGeneralizedCoordinates(), states[i].GetVelocity(),
            states[i].GetAcceleration(), states[i].GetAlgorithmicAcceleration()};
        log->Info("** Integrating step number " + std::to_string(i + 1) + " **\n");
        states.emplace_back(
            std::get<0>(this->AlphaStep(input_state, n_constraints, linearization_parameters))
        );
    }
    log->Info("Time integration has completed!\n");
    return states;
}

std::tuple<State, Kokkos::View<double*>> GeneralizedAlphaTimeIntegrator::AlphaStep(
    const State& state, size_t n_constraints,
    std::shared_ptr<LinearizationParameters> linearization_parameters
) {
    auto gen_coords = state.GetGeneralizedCoordinates();
    auto velocity = state.GetVelocity();
    auto acceleration = state.GetAcceleration();
    auto algo_acceleration = state.GetAlgorithmicAcceleration();

    // Define some constants that will be used in the algorithm
    const auto size_dofs = velocity.extent(0) * velocity.extent(1);
    const auto size_constraints = n_constraints;
    const auto size_problem = size_dofs + size_constraints;

    const auto h = this->time_stepper_.GetTimeStep();
    const auto n_nodes = state.GetNumberOfNodes();

    const auto kAlphaFLocal = kAlphaF_;
    const auto kAlphaMLocal = kAlphaM_;
    const auto kBetaLocal = kBeta_;
    const auto kGammaLocal = kGamma_;

    // Initialize some X_next variables to assist in updating the State
    auto gen_coords_next =
        Kokkos::View<double* [kNumberOfLieGroupComponents]>("gen_coords_next", n_nodes);
    auto velocity_next =
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]>("velocity_next", n_nodes);
    auto acceleration_next =
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]>("acceleration_next", n_nodes);
    auto algo_acceleration_next =
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]>("algo_acceleration_next", n_nodes);
    auto delta_gen_coords =
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]>("delta_gen_coords", n_nodes);
    auto lagrange_mults_next = Kokkos::View<double*>("lagrange_mults_next", n_constraints);

    // Loop over all nodes in the system and update the generalized coordinates, velocities,
    // accelerations, and algorithmic accelerations
    auto log = util::Log::Get();
    log->Info(
        "Performing Newton-Raphson iterations to update solution using the generalized-alpha "
        "algorithm\n"
    );

    for (size_t node = 0; node < n_nodes; node++) {
        // Perform the linear update part of the generalized alpha algorithm
        // Algorithm from Table 1, Brüls, Cardona, and Arnold 2012
        Kokkos::parallel_for(
            kNumberOfLieAlgebraComponents,
            KOKKOS_LAMBDA(const size_t i) {
                algo_acceleration_next(node, i) = (kAlphaFLocal * acceleration(node, i) -
                                                   kAlphaMLocal * algo_acceleration(node, i)) /
                                                  (1. - kAlphaMLocal);

                delta_gen_coords(node, i) = velocity(node, i) +
                                            h * (0.5 - kBetaLocal) * algo_acceleration(node, i) +
                                            h * kBetaLocal * algo_acceleration_next(node, i);

                velocity_next(node, i) = velocity(node, i) +
                                         h * (1 - kGammaLocal) * algo_acceleration(node, i) +
                                         h * kGammaLocal * algo_acceleration_next(node, i);

                algo_acceleration(node, i) = algo_acceleration_next(node, i);

                acceleration_next(node, i) = 0.;
            }
        );
    }

    // Initialize lagrange_mults_next to zero separately since it might be of different size
    Kokkos::deep_copy(lagrange_mults_next, 0.);

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha
    // algorithm
    const auto kBetaPrime = (1 - kAlphaM_) / (h * h * kBeta_ * (1 - kAlphaF_));
    const auto kGammaPrime = kGamma_ / (h * kBeta_);

    // Precondition the linear solve (Bottasso et al 2008)
    const auto dl = Kokkos::View<double**>("dl", size_problem, size_problem);
    const auto dr = Kokkos::View<double**>("dr", size_problem, size_problem);
    Kokkos::deep_copy(dl, 0.);
    Kokkos::deep_copy(dr, 0.);

    if (this->is_preconditioned_) {
        Kokkos::parallel_for(
            size_problem,
            KOKKOS_LAMBDA(const size_t i) {
                dl(i, i) = 1.;
                dr(i, i) = 1.;
            }
        );

        Kokkos::parallel_for(
            size_problem,
            KOKKOS_LAMBDA(const size_t i) {
                if (i >= size_dofs) {
                    dr(i, i) = 1. / (kBetaLocal * h * h);
                } else {
                    dl(i, i) = kBetaLocal * h * h;
                }
            }
        );
    }

    // Allocate some Views to assist in performing the Newton-Raphson iterations
    auto residuals = Kokkos::View<double*>("residuals", size_problem);
    auto iteration_matrix = Kokkos::View<double**>("iteration_matrix", size_problem, size_problem);
    auto soln_increments = Kokkos::View<double*>("soln_increments", size_problem);
    auto delta_x = Kokkos::View<double*>("delta_x", size_problem);
    auto delta_lagrange_mults = Kokkos::View<double*>("delta_lagrange_mults", size_constraints);

    const auto max_iterations = this->time_stepper_.GetMaximumNumberOfIterations();
    for (time_stepper_.SetNumberOfIterations(0);
         time_stepper_.GetNumberOfIterations() < max_iterations;
         time_stepper_.IncrementNumberOfIterations()) {
        log->Debug(
            "Performing Newton-Raphson iteration number " +
            std::to_string(time_stepper_.GetNumberOfIterations() + 1) + "\n"
        );

        UpdateGeneralizedCoordinates(gen_coords, delta_gen_coords, gen_coords_next);

        // Compute the residuals and check for convergence
        linearization_parameters->ResidualVector(
            gen_coords_next, velocity_next, acceleration_next, lagrange_mults_next, residuals
        );

        if (this->IsConverged(residuals)) {
            this->is_converged_ = true;
            break;
        }

        linearization_parameters->IterationMatrix(
            h, kBetaPrime, kGammaPrime, gen_coords_next, delta_gen_coords, velocity_next,
            acceleration_next, lagrange_mults_next, iteration_matrix
        );

        if (this->is_preconditioned_) {
            iteration_matrix = gen_alpha_solver::multiply_matrix_with_matrix(iteration_matrix, dr);
            iteration_matrix = gen_alpha_solver::multiply_matrix_with_matrix(dl, iteration_matrix);

            Kokkos::parallel_for(
                size_dofs,
                KOKKOS_LAMBDA(const size_t i) { residuals(i) = residuals(i) * kBetaLocal * h * h; }
            );
        }

        Kokkos::deep_copy(soln_increments, residuals);
        openturbine::gebt_poc::solve_linear_system(iteration_matrix, soln_increments);

        if (this->problem_type_ == ProblemType::kDynamic) {
            // Check for convergence based on energy criterion
            if (this->IsConverged(residuals, soln_increments)) {
                this->is_converged_ = true;
                break;
            }
        }

        Kokkos::parallel_for(
            delta_gen_coords.size(),
            // Take negative of the solution increments to update generalized coordinates
            KOKKOS_LAMBDA(const size_t i) { delta_x(i) = -soln_increments(i); }
        );

        if (n_constraints > 0) {
            if (this->is_preconditioned_) {
                Kokkos::parallel_for(
                    n_constraints,
                    KOKKOS_LAMBDA(const size_t i) {
                        // Take negative of the solution increments to update Lagrange
                        // multipliers
                        delta_lagrange_mults(i) =
                            -soln_increments(i + delta_gen_coords.size()) / (kBetaLocal * h * h);
                        lagrange_mults_next(i) += delta_lagrange_mults(i);
                    }
                );
            } else {
                Kokkos::parallel_for(
                    n_constraints,
                    KOKKOS_LAMBDA(const size_t i) {
                        // Take negative of the solution increments to update Lagrange
                        // multipliers
                        delta_lagrange_mults(i) = -soln_increments(i + delta_gen_coords.size());
                        lagrange_mults_next(i) += delta_lagrange_mults(i);
                    }
                );
            }
        }

        for (size_t node = 0; node < n_nodes; node++) {
            // Update the velocity, acceleration, and constraints based on the increments
            Kokkos::parallel_for(
                kNumberOfLieAlgebraComponents,
                KOKKOS_LAMBDA(const size_t i) {
                    delta_gen_coords(node, i) += delta_x(node * kNumberOfLieAlgebraComponents + i);
                    velocity_next(node, i) +=
                        kGammaPrime * delta_x(node * kNumberOfLieAlgebraComponents + i);
                    acceleration_next(node, i) +=
                        kBetaPrime * delta_x(node * kNumberOfLieAlgebraComponents + i);
                }
            );
        }
    }

    // Update algorithmic acceleration once Newton-Raphson iterations have ended
    for (size_t node = 0; node < n_nodes; node++) {
        Kokkos::parallel_for(
            kNumberOfLieAlgebraComponents,
            KOKKOS_LAMBDA(const size_t i) {
                algo_acceleration_next(node, i) +=
                    (1. - kAlphaFLocal) / (1. - kAlphaMLocal) * acceleration_next(node, i);
            }
        );
    }

    const auto n_iterations = time_stepper_.GetNumberOfIterations();
    this->time_stepper_.IncrementTotalNumberOfIterations(n_iterations);

    auto results = std::make_tuple(
        State{gen_coords_next, velocity_next, acceleration_next, algo_acceleration_next},
        lagrange_mults_next
    );

    if (this->is_converged_) {
        log->Info(
            "Newton-Raphson iterations converged in " + std::to_string(n_iterations) +
            " iterations\n"
        );
        return results;
    }

    log->Warning(
        "Newton-Raphson iterations failed to converge on a solution after " +
        std::to_string(n_iterations) + " iterations!\n"
    );

    return results;
}

void GeneralizedAlphaTimeIntegrator::UpdateGeneralizedCoordinates(
    Kokkos::View<const double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<const double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords_next
) {
    const auto h = this->time_stepper_.GetTimeStep();

    // Loop over all nodes in the system and update the generalized coordinates
    for (size_t node = 0; node < gen_coords.extent(0); node++) {
        auto update_generalized_coordinates = KOKKOS_LAMBDA(size_t) {
            // {gen_coords_next} = {gen_coords} + h * {delta_gen_coords}
            //
            // Step 1: R^3 update, done with vector addition
            auto current_position = gen_alpha_solver::Vector{
                gen_coords(node, 0), gen_coords(node, 1), gen_coords(node, 2)};
            auto updated_position = gen_alpha_solver::Vector{
                delta_gen_coords(node, 0), delta_gen_coords(node, 1), delta_gen_coords(node, 2)};
            auto r = current_position + (updated_position * h);

            // Step 2: SO(3) update, done with quaternion composition
            gen_alpha_solver::Quaternion current_orientation{
                gen_coords(node, 3), gen_coords(node, 4), gen_coords(node, 5), gen_coords(node, 6)};
            auto updated_orientation = gen_alpha_solver::quaternion_from_rotation_vector(
                // Convert Vector -> Quaternion via exponential mapping
                gen_alpha_solver::Vector{
                    delta_gen_coords(node, 3), delta_gen_coords(node, 4),
                    delta_gen_coords(node, 5)} *
                h
            );
            auto q = current_orientation * updated_orientation;

            gen_coords_next(node, 0) = r.GetXComponent();
            gen_coords_next(node, 1) = r.GetYComponent();
            gen_coords_next(node, 2) = r.GetZComponent();
            gen_coords_next(node, 3) = q.GetScalarComponent();
            gen_coords_next(node, 4) = q.GetXComponent();
            gen_coords_next(node, 5) = q.GetYComponent();
            gen_coords_next(node, 6) = q.GetZComponent();
        };
        Kokkos::parallel_for(1, update_generalized_coordinates);
    }
}

bool GeneralizedAlphaTimeIntegrator::IsConverged(const Kokkos::View<double*> residual) {
    // L2 norm of the residual vector should be very small (< epsilon) for the solution
    // to be considered converged
    auto log = util::Log::Get();
    log->Debug("Norm of residual: " + std::to_string(KokkosBlas::nrm2(residual)) + "\n");
    return KokkosBlas::nrm2(residual) < kConvergenceTolerance;
}

bool GeneralizedAlphaTimeIntegrator::IsConverged(
    Kokkos::View<double*> residual, Kokkos::View<double*> solution_increment
) {
    auto energy_increment = std::abs(KokkosBlas::dot(residual, solution_increment));

    // Store the first energy increment as the reference energy
    if (this->time_stepper_.GetNumberOfIterations() == 1) {
        this->reference_energy_ = energy_increment;
    }
    auto energy_ratio = energy_increment / reference_energy_;

    auto log = util::Log::Get();
    log->Debug(
        "Energy increment: " + std::to_string(energy_increment) + "\n" +
        "Energy ratio: " + std::to_string(energy_ratio) + "\n"
    );

    if (energy_increment < 1e-8 || energy_ratio < 1e-5) {
        log->Debug("Solution converged for dynamic problem!\n");
        return true;
    }
    return false;
}

}  // namespace openturbine::gebt_poc
