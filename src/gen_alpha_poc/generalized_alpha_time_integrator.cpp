// #include "src/gen_alpha_poc/generalized_alpha_time_integrator.h"

// #include "src/gen_alpha_poc/heavy_top.h"
// #include "src/gen_alpha_poc/quaternion.h"
// #include "src/gen_alpha_poc/solver.h"
// #include "src/gen_alpha_poc/vector.h"
// #include "src/utilities/log.h"

// namespace openturbine::gen_alpha_solver {

// GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
//     double alpha_f, double alpha_m, double beta, double gamma, TimeStepper time_stepper,
//     bool precondition
// )
//     : kAlphaF_(alpha_f),
//       kAlphaM_(alpha_m),
//       kBeta_(beta),
//       kGamma_(gamma),
//       time_stepper_(std::move(time_stepper)),
//       is_preconditioned_(precondition) {
//     if (this->kAlphaF_ < 0 || this->kAlphaF_ > 1) {
//         throw std::invalid_argument("Invalid value provided for alpha_f");
//     }
//     if (this->kAlphaM_ < 0 || this->kAlphaM_ > 1) {
//         throw std::invalid_argument("Invalid value provided for alpha_m");
//     }
//     if (this->kBeta_ < 0 || this->kBeta_ > 0.50) {
//         throw std::invalid_argument("Invalid value provided for beta");
//     }
//     if (this->kGamma_ < 0 || this->kGamma_ > 1) {
//         throw std::invalid_argument("Invalid value provided for gamma");
//     }
//     this->is_converged_ = false;
// }

// std::vector<State> GeneralizedAlphaTimeIntegrator::Integrate(
//     const State& initial_state, size_t n_constraints,
//     std::shared_ptr<LinearizationParameters> linearization_parameters
// ) {
//     auto n_gen_coords = initial_state.GetGeneralizedCoordinates().size();
//     auto n_velocities = initial_state.GetVelocity().size();
//     auto n_accelerations = initial_state.GetAcceleration().size();
//     auto n_algo_accelerations = initial_state.GetAlgorithmicAcceleration().size();

//     if (n_velocities != kNumberOfLieAlgebraComponents ||
//         n_accelerations != kNumberOfLieAlgebraComponents ||
//         n_algo_accelerations != kNumberOfLieAlgebraComponents) {
//         throw std::invalid_argument(
//             "The number of velocities, accelerations, and algorithmic accelerations in the "
//             "initial state must be of size 6 or its multiple for the lie group based generalized "
//             "alpha integrator"
//         );
//     }

//     if (n_gen_coords != kNumberOfLieGroupComponents) {
//         throw std::invalid_argument(
//             "The number of generalized coordinates in the initial state must be of size "
//             "7 or its multiple for the lie group based generalized alpha integrator"
//         );
//     }

//     auto log = util::Log::Get();
//     std::vector<State> states{initial_state};
//     auto n_steps = this->time_stepper_.GetNumberOfSteps();
//     for (size_t i = 0; i < n_steps; i++) {
//         this->time_stepper_.AdvanceTimeStep();
//         auto input_state = State(
//             states[i].GetGeneralizedCoordinates(), states[i].GetVelocity(),
//             states[i].GetAcceleration(), states[i].GetAlgorithmicAcceleration()
//         );
//         log->Info("** Integrating step number " + std::to_string(i + 1) + " **\n");
//         states.emplace_back(
//             std::get<0>(this->AlphaStep(input_state, n_constraints, linearization_parameters))
//         );
//     }
//     log->Info("Time integration has completed!\n");
//     return states;
// }

// std::tuple<State, Kokkos::View<double*>> GeneralizedAlphaTimeIntegrator::AlphaStep(
//     const State& state, size_t n_constraints,
//     std::shared_ptr<LinearizationParameters> linearization_parameters
// ) {
//     auto gen_coords = state.GetGeneralizedCoordinates();
//     auto velocity = state.GetVelocity();
//     auto acceleration = state.GetAcceleration();
//     auto algo_acceleration = state.GetAlgorithmicAcceleration();

//     // Initialize some X_next variables to assist in updating the State - only for ones that
//     // require both the current and next values in a calculation
//     auto gen_coords_next = Kokkos::View<double*>("gen_coords_next", gen_coords.size());
//     auto algo_acceleration_next =
//         Kokkos::View<double*>("algorithmic_acceleration_next", algo_acceleration.size());
//     auto delta_gen_coords = Kokkos::View<double*>("gen_coords_increment", velocity.size());
//     auto lagrange_mults_next = Kokkos::View<double*>("lagrange_mults_next", n_constraints);

//     // Perform the linear update part of the generalized alpha algorithm
//     const auto h = this->time_stepper_.GetTimeStep();
//     const auto size = velocity.size();

//     const auto kAlphaFLocal = kAlphaF_;
//     const auto kAlphaMLocal = kAlphaM_;
//     const auto kBetaLocal = kBeta_;
//     const auto kGammaLocal = kGamma_;

//     // Algorithm from Table 1, Brüls, Cardona, and Arnold 2012
//     Kokkos::parallel_for(
//         size,
//         KOKKOS_LAMBDA(const size_t i) {
//             algo_acceleration_next(i) =
//                 (kAlphaFLocal * acceleration(i) - kAlphaMLocal * algo_acceleration(i)) /
//                 (1. - kAlphaMLocal);

//             delta_gen_coords(i) = velocity(i) + h * (0.5 - kBetaLocal) * algo_acceleration(i) +
//                                   h * kBetaLocal * algo_acceleration_next(i);
//             velocity(i) += h * (1 - kGammaLocal) * algo_acceleration(i) +
//                            h * kGammaLocal * algo_acceleration_next(i);

//             algo_acceleration(i) = algo_acceleration_next(i);

//             acceleration(i) = 0.;
//         }
//     );

//     // Initialize lagrange_mults_next to zero separately since it might be of different size
//     Kokkos::deep_copy(lagrange_mults_next, 0.);

//     // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha algorithm
//     auto log = util::Log::Get();
//     log->Info(
//         "Performing Newton-Raphson iterations to update solution using the generalized-alpha "
//         "algorithm\n"
//     );

//     const auto kBetaPrime = (1 - kAlphaM_) / (h * h * kBeta_ * (1 - kAlphaF_));
//     const auto kGammaPrime = kGamma_ / (h * kBeta_);

//     // Precondition the linear solve (Bottasso et al 2008)
//     const auto dl = Kokkos::View<double**>("dl", size + n_constraints, size + n_constraints);
//     const auto dr = Kokkos::View<double**>("dr", size + n_constraints, size + n_constraints);
//     Kokkos::deep_copy(dl, 0.);
//     Kokkos::deep_copy(dr, 0.);

//     if (this->is_preconditioned_) {
//         Kokkos::parallel_for(
//             size + n_constraints,
//             KOKKOS_LAMBDA(const size_t i) {
//                 dl(i, i) = 1.;
//                 dr(i, i) = 1.;
//             }
//         );

//         Kokkos::parallel_for(
//             size + n_constraints,
//             KOKKOS_LAMBDA(const size_t i) {
//                 if (i >= size) {
//                     dr(i, i) = 1. / (kBetaLocal * h * h);
//                 } else {
//                     dl(i, i) = kBetaLocal * h * h;
//                 }
//             }
//         );
//     }

//     const auto max_iterations = this->time_stepper_.GetMaximumNumberOfIterations();
//     for (time_stepper_.SetNumberOfIterations(0);
//          time_stepper_.GetNumberOfIterations() < max_iterations;
//          time_stepper_.IncrementNumberOfIterations()) {
//         gen_coords_next = UpdateGeneralizedCoordinates(gen_coords, delta_gen_coords);

//         // Compute the residuals and check for convergence
//         const auto residuals = linearization_parameters->ResidualVector(
//             gen_coords_next, velocity, acceleration, lagrange_mults_next
//         );

//         if (this->CheckConvergence(residuals)) {
//             this->is_converged_ = true;
//             break;
//         }

//         auto iteration_matrix = linearization_parameters->IterationMatrix(
//             h, kBetaPrime, kGammaPrime, gen_coords_next, delta_gen_coords, velocity, acceleration,
//             lagrange_mults_next
//         );

//         if (this->is_preconditioned_) {
//             iteration_matrix = multiply_matrix_with_matrix(iteration_matrix, dr);
//             iteration_matrix = multiply_matrix_with_matrix(dl, iteration_matrix);

//             Kokkos::parallel_for(
//                 size,
//                 KOKKOS_LAMBDA(const size_t i) { residuals(i) = residuals(i) * kBetaLocal * h * h;
//                 }
//             );
//         }

//         auto soln_increments = Kokkos::View<double*>("soln_increments", residuals.size());
//         Kokkos::deep_copy(soln_increments, residuals);
//         solve_linear_system(iteration_matrix, soln_increments);

//         Kokkos::View<double*> delta_x("delta_x", delta_gen_coords.size());
//         Kokkos::parallel_for(
//             delta_gen_coords.size(),
//             // Take negative of the solution increments to update generalized coordinates
//             KOKKOS_LAMBDA(const size_t i) { delta_x(i) = -soln_increments(i); }
//         );

//         if (n_constraints > 0) {
//             Kokkos::View<double*> delta_lagrange_mults("delta_lagrange_mults", n_constraints);

//             if (this->is_preconditioned_) {
//                 Kokkos::parallel_for(
//                     n_constraints,
//                     KOKKOS_LAMBDA(const size_t i) {
//                         // Take negative of the solution increments to update Lagrange multipliers
//                         delta_lagrange_mults(i) =
//                             -soln_increments(i + delta_gen_coords.size()) / (kBetaLocal * h * h);
//                         lagrange_mults_next(i) += delta_lagrange_mults(i);
//                     }
//                 );
//             } else {
//                 Kokkos::parallel_for(
//                     n_constraints,
//                     KOKKOS_LAMBDA(const size_t i) {
//                         // Take negative of the solution increments to update Lagrange multipliers
//                         delta_lagrange_mults(i) = -soln_increments(i + delta_gen_coords.size());
//                         lagrange_mults_next(i) += delta_lagrange_mults(i);
//                     }
//                 );
//             }
//         }

//         // Update the velocity, acceleration, and constraints based on the increments
//         Kokkos::parallel_for(
//             size,
//             KOKKOS_LAMBDA(const size_t i) {
//                 delta_gen_coords(i) += delta_x(i) / h;
//                 velocity(i) += kGammaPrime * delta_x(i);
//                 acceleration(i) += kBetaPrime * delta_x(i);
//             }
//         );
//     }

//     const auto n_iterations = time_stepper_.GetNumberOfIterations();
//     this->time_stepper_.IncrementTotalNumberOfIterations(n_iterations);

//     // Update algorithmic acceleration once Newton-Raphson iterations have ended
//     Kokkos::parallel_for(
//         size,
//         KOKKOS_LAMBDA(const size_t i) {
//             algo_acceleration_next(i) += (1. - kAlphaFLocal) / (1. - kAlphaMLocal) *
//             acceleration(i);
//         }
//     );

//     auto results = std::make_tuple(
//         State{gen_coords_next, velocity, acceleration, algo_acceleration_next},
//         lagrange_mults_next
//     );

//     if (this->is_converged_) {
//         log->Info(
//             "Newton-Raphson iterations converged in " + std::to_string(n_iterations + 1) +
//             " iterations\n"
//         );
//         return results;
//     }

//     log->Warning(
//         "Newton-Raphson iterations failed to converge on a solution after " +
//         std::to_string(n_iterations + 1) + " iterations!\n"
//     );
//     return results;
// }

// Kokkos::View<double*> GeneralizedAlphaTimeIntegrator::UpdateGeneralizedCoordinates(
//     const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> delta_gen_coords
// ) {
//     auto gen_coords_next = Kokkos::View<double*>("generalized_coordinates_next",
//     gen_coords.size()); const auto h = this->time_stepper_.GetTimeStep();

//     auto update_generalized_coordinates = KOKKOS_LAMBDA(size_t) {
//         // {gen_coords_next} = {gen_coords} + h * {delta_gen_coords}
//         //
//         // Step 1: R^3 update, done with vector addition
//         auto current_position = Vector{gen_coords(0), gen_coords(1), gen_coords(2)};
//         auto updated_position =
//             Vector{delta_gen_coords(0), delta_gen_coords(1), delta_gen_coords(2)};
//         auto r = current_position + (updated_position * h);

//         // Step 2: SO(3) update, done with quaternion composition
//         Quaternion current_orientation{gen_coords(3), gen_coords(4), gen_coords(5),
//         gen_coords(6)}; auto updated_orientation = quaternion_from_rotation_vector(
//             // Convert Vector -> Quaternion via exponential mapping
//             Vector{delta_gen_coords(3), delta_gen_coords(4), delta_gen_coords(5)} * h
//         );
//         auto q = current_orientation * updated_orientation;

//         gen_coords_next(0) = r.GetXComponent();
//         gen_coords_next(1) = r.GetYComponent();
//         gen_coords_next(2) = r.GetZComponent();
//         gen_coords_next(3) = q.GetScalarComponent();
//         gen_coords_next(4) = q.GetXComponent();
//         gen_coords_next(5) = q.GetYComponent();
//         gen_coords_next(6) = q.GetZComponent();
//     };

//     Kokkos::parallel_for(1, update_generalized_coordinates);

//     return gen_coords_next;
// }

// bool GeneralizedAlphaTimeIntegrator::CheckConvergence(const Kokkos::View<double*> residual) {
//     // L2 norm of the residual vector should be very small (< epsilon) for the solution
//     // to be considered converged
//     double residual_norm = 0.;
//     Kokkos::parallel_reduce(
//         residual.extent(0),
//         KOKKOS_LAMBDA(int i, double& residual_partial_sum) {
//             double residual_value = residual(i);
//             residual_partial_sum += residual_value * residual_value;
//         },
//         Kokkos::Sum<double>(residual_norm)
//     );
//     residual_norm = std::sqrt(residual_norm);

//     return (residual_norm) < kConvergenceTolerance ? true : false;
// }

// }  // namespace openturbine::gen_alpha_solver
