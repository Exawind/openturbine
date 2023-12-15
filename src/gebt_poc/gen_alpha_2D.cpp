#include "src/gebt_poc/gen_alpha_2D.h"

#include <KokkosBlas.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>

#include "src/gebt_poc/linear_solver.h"
#include "src/gen_alpha_poc/heavy_top.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/vector.h"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

GeneralizedAlphaTimeIntegrator::GeneralizedAlphaTimeIntegrator(
    double alpha_f, double alpha_m, double beta, double gamma,
    gen_alpha_solver::TimeStepper time_stepper, bool precondition
)
    : kAlphaF_(alpha_f),
      kAlphaM_(alpha_m),
      kBeta_(beta),
      kGamma_(gamma),
      time_stepper_(std::move(time_stepper)),
      is_preconditioned_(precondition) {
    if (this->kAlphaF_ < 0 || this->kAlphaF_ > 1) {
        throw std::invalid_argument("Invalid value provided for alpha_f");
    }
    if (this->kAlphaM_ < 0 || this->kAlphaM_ > 1) {
        throw std::invalid_argument("Invalid value provided for alpha_m");
    }
    if (this->kBeta_ < 0 || this->kBeta_ > 0.50) {
        throw std::invalid_argument("Invalid value provided for beta");
    }
    if (this->kGamma_ < 0 || this->kGamma_ > 1) {
        throw std::invalid_argument("Invalid value provided for gamma");
    }
    this->is_converged_ = false;
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

template<typename M1, typename M2, typename M3>
void ApplyPreconditioner(M1 A, M2 R, M3 L) {
  using member_type = Kokkos::TeamPolicy<>::member_type;
  using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  using matrix = Kokkos::View<double**, scratch_space, unmanaged_memory>;
  using no_transpose = KokkosBatched::Trans::NoTranspose;
  using unblocked = KokkosBatched::Algo::Gemm::Unblocked;
  using gemm = KokkosBatched::TeamVectorGemm<member_type, no_transpose, no_transpose, unblocked>;
  auto policy = Kokkos::TeamPolicy<>(1, Kokkos::AUTO(), Kokkos::AUTO());
  auto n = A.extent(0);
  auto m = A.extent(1);
  auto scratch_size = matrix::shmem_size(n, m);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const member_type& member) { 
    auto tmp = matrix(member.team_scratch(0), n, m);
    gemm::invoke(member, 1., A, R, 0., tmp);
    gemm::invoke(member, 1., L, tmp, 0., A);
  });
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
    using LieGroupField = Kokkos::View<double* [kNumberOfLieGroupComponents]>;
    using LieAlgebraField = Kokkos::View<double* [kNumberOfLieAlgebraComponents]>;
    auto gen_coords_next = LieGroupField("gen_coords_next", n_nodes);
    auto velocity_next = LieAlgebraField("velocity_next", n_nodes);
    auto acceleration_next = LieAlgebraField("acceleration_next", n_nodes);
    auto algo_acceleration_next = LieAlgebraField("algo_acceleration_next", n_nodes);
    auto delta_gen_coords = LieAlgebraField("delta_gen_coords", n_nodes);
    auto lagrange_mults_next = Kokkos::View<double*>("lagrange_mults_next", n_constraints);

    // Loop over all nodes in the system and update the generalized coordinates, velocities,
    // accelerations, and algorithmic accelerations
    auto log = util::Log::Get();
    log->Info(
        "Performing Newton-Raphson iterations to update solution using the generalized-alpha "
        "algorithm\n"
    );

    using member_type = Kokkos::TeamPolicy<>::member_type;
    auto node_team_policy = Kokkos::TeamPolicy<>(n_nodes, Kokkos::AUTO(), Kokkos::AUTO());
    Kokkos::parallel_for(node_team_policy, KOKKOS_LAMBDA(const member_type& member) {
      auto node = member.league_rank();
      constexpr auto components = kNumberOfLieAlgebraComponents;
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
          algo_acceleration_next(node, i) = (kAlphaFLocal * acceleration(node, i) -
                                             kAlphaMLocal * algo_acceleration(node, i)) /
                                            (1. - kAlphaMLocal);
        });
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
                delta_gen_coords(node, i) = velocity(node, i) +
                                            h * (0.5 - kBetaLocal) * algo_acceleration(node, i) +
                                            h * kBetaLocal * algo_acceleration_next(node, i);
        });
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
                velocity_next(node, i) = velocity(node, i) +
                                         h * (1 - kGammaLocal) * algo_acceleration(node, i) +
                                         h * kGammaLocal * algo_acceleration_next(node, i);
        });
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
                algo_acceleration(node, i) = algo_acceleration_next(node, i);
        });
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
                acceleration_next(node, i) = 0.;
        });
    });

    // Initialize lagrange_mults_next to zero separately since it might be of different size
    Kokkos::deep_copy(lagrange_mults_next, 0.);

    // Perform Newton-Raphson iterations to update nonlinear part of generalized-alpha
    // algorithm
    const auto kBetaPrime = (1 - kAlphaM_) / (h * h * kBeta_ * (1 - kAlphaF_));
    const auto kGammaPrime = kGamma_ / (h * kBeta_);

    // Precondition the linear solve (Bottasso et al 2008)
    auto preconditioning_factor = (is_preconditioned_) ? (kBetaLocal * h * h) : 1.;
    auto dl = Kokkos::View<double**>("dl", size_problem, size_problem);
    auto dr = Kokkos::View<double**>("dr", size_problem, size_problem);

        Kokkos::parallel_for(size_problem, KOKKOS_LAMBDA(const size_t i) {
                dl(i, i) = (i < size_dofs) ? preconditioning_factor : 1.;
                dr(i, i) = (i >= size_dofs) ? 1. / preconditioning_factor : 1.;
            }
        );

    // Allocate some Views to assist in performing the Newton-Raphson iterations
    auto residuals = Kokkos::View<double*>("residuals", size_problem);
    auto iteration_matrix = Kokkos::View<double**>("iteration_matrix", size_problem, size_problem);

    const auto max_iterations = this->time_stepper_.GetMaximumNumberOfIterations();
    for (time_stepper_.SetNumberOfIterations(0);
         time_stepper_.GetNumberOfIterations() < max_iterations;
         time_stepper_.IncrementNumberOfIterations()) {
        log->Debug(
            "Performing Newton-Raphson iteration number " +
            std::to_string(time_stepper_.GetNumberOfIterations() + 1) + "\n"
        );

        UpdateGeneralizedCoordinates(gen_coords, delta_gen_coords, gen_coords_next);

        linearization_parameters->ResidualVector(
            gen_coords_next, velocity_next, acceleration_next, lagrange_mults_next, residuals
        );

        if (is_converged_ = IsConverged(residuals)) {
            break;
        }

        linearization_parameters->IterationMatrix(
            h, kBetaPrime, kGammaPrime, gen_coords_next, delta_gen_coords, velocity_next,
            acceleration_next, lagrange_mults_next, iteration_matrix
        );

        // Solve the Linear System
        ApplyPreconditioner(iteration_matrix, dr, dl);
        auto solution_residual = Kokkos::subview(residuals, Kokkos::make_pair(0ul, size_dofs));
        KokkosBlas::scal(solution_residual, preconditioning_factor, solution_residual);
        openturbine::gebt_poc::solve_linear_system(iteration_matrix, residuals);
        KokkosBlas::scal(residuals, -1, residuals);

        // Update constraints based on the increments
        auto delta_lagrange_mults = Kokkos::subview(residuals, Kokkos::make_pair(size_dofs, size_problem));
        KokkosBlas::axpy(1. / preconditioning_factor, delta_lagrange_mults, lagrange_mults_next);

        // Update states based on the increments
        Kokkos::parallel_for(node_team_policy, KOKKOS_LAMBDA(const member_type& member) {
          auto node = member.league_rank();
          constexpr auto components = kNumberOfLieAlgebraComponents;
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
            delta_gen_coords(node, i) += residuals(node * components + i);
          });
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
            velocity_next(node, i) += kGammaPrime * residuals(node * components + i);
          });
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
            acceleration_next(node, i) += kBetaPrime * residuals(node * components + i);
          });
        });
    }

    // Update algorithmic acceleration once Newton-Raphson iterations have ended
    Kokkos::parallel_for(node_team_policy, KOKKOS_LAMBDA(const member_type& member) {
      auto node = member.league_rank();
      constexpr auto components = kNumberOfLieAlgebraComponents;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, components), [=](std::size_t i) {
        algo_acceleration_next(node, i) += (1. - kAlphaFLocal) / (1. - kAlphaMLocal) * acceleration_next(node, i);
      });
    });

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

template<typename MemberType, typename QuaternionView, typename VectorView>
KOKKOS_FUNCTION
void compute_orientation_quaternion(MemberType& member, QuaternionView orientation, VectorView vector, double h) {
    auto angle = KokkosBlas::serial_nrm2(vector) * h;
    auto is_small_angle = std::abs(angle) < 1.e-6;
    auto factor = h * std::sin(angle / 2.0) / angle;
    Kokkos::single(Kokkos::PerTeam(member), [=]() {
      orientation(0) = (is_small_angle) ? 1. : std::cos(angle / 2.0);
    });
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 3), [=](std::size_t i) {
      orientation(i+1) = (is_small_angle) ? 0. : vector(i) * factor;
    });
}

template<typename MemberType, typename QOut, typename Q1, typename Q2>
KOKKOS_FUNCTION
void compose_quaternions(MemberType& member, QOut q_out, Q1 q1, Q2 q2) {
  Kokkos::single(Kokkos::PerTeam(member), [=]() {
        q_out(0) = q1(0) * q2(0) - q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3);
        q_out(1) = q1(0) * q2(1) + q1(1) * q2(0) + q1(2) * q2(3) - q1(3) * q2(2);
        q_out(2) = q1(0) * q2(2) - q1(1) * q2(3) + q1(2) * q2(0) + q1(3) * q2(1);
        q_out(3) = q1(0) * q2(3) + q1(1) * q2(2) - q1(2) * q2(1) + q1(3) * q2(0);
  });
}

void GeneralizedAlphaTimeIntegrator::UpdateGeneralizedCoordinates(
    Kokkos::View<const double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<const double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords_next
) {
    const auto h = this->time_stepper_.GetTimeStep();
    const auto n_nodes = gen_coords.extent(0);
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using vector = Kokkos::View<double[4], scratch_space, unmanaged_memory>;
    using member_type = Kokkos::TeamPolicy<>::member_type;
    auto node_team_policy = Kokkos::TeamPolicy<>(n_nodes, Kokkos::AUTO(), Kokkos::AUTO());
    auto scratch_size = Kokkos::View<double[4]>::shmem_size();
    node_team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    Kokkos::parallel_for(node_team_policy, KOKKOS_LAMBDA(const member_type& member) {
      auto node = member.league_rank();

      //R^3 Update
      auto current_position = Kokkos::subview(gen_coords, node, Kokkos::make_pair(0, 3));
      auto updated_position = Kokkos::subview(delta_gen_coords, node, Kokkos::make_pair(0, 3));
      auto r = Kokkos::subview(gen_coords_next, node, Kokkos::make_pair(0, 3));

      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 3), [=](std::size_t i) {
        r(i) = current_position(i) + updated_position(i) * h;
      });

      //SO(3) Update
      auto current_orientation = Kokkos::subview(gen_coords, node, Kokkos::make_pair(3, 7));
      auto updated_orientation_vector = Kokkos::subview(delta_gen_coords, node, Kokkos::make_pair(3, 6));
      auto updated_orientation = vector(member.team_scratch(0));
      auto q = Kokkos::subview(gen_coords_next, node, Kokkos::make_pair(3, 7));

      compute_orientation_quaternion(member, updated_orientation, updated_orientation_vector, h);
      compose_quaternions(member, q, current_orientation, updated_orientation);
    });
}

bool GeneralizedAlphaTimeIntegrator::IsConverged(const Kokkos::View<double*> residual) {
    // L2 norm of the residual vector should be very small (< epsilon) for the solution
    // to be considered converged
    return KokkosBlas::nrm2(residual) < kConvergenceTolerance;
}

}  // namespace openturbine::gebt_poc
