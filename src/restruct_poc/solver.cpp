#include "solver.hpp"

#include "beams.hpp"
#include "solver_functors.hpp"

#include "src/gebt_poc/linear_solver.h"

namespace openturbine {

void PredictNextState(Solver& solver) {
    Kokkos::deep_copy(solver.state.lambda, 0.);
    Kokkos::deep_copy(solver.state.q_prev, solver.state.q);

    // Predict the new state values
    Kokkos::parallel_for(
        "CalculateNextState", solver.num_system_nodes,
        CalculateNextState{
            solver.h, solver.alpha_f, solver.alpha_m, solver.beta, solver.gamma,
            solver.state.q_delta, solver.state.v, solver.state.vd, solver.state.a}
    );

    // Update predicted displacements
    Kokkos::parallel_for(
        "CalculateDisplacement", solver.state.q.extent(0),
        CalculateDisplacement{solver.h, solver.state.q_delta, solver.state.q_prev, solver.state.q}
    );
}

void InitializeConstraints(Solver& solver, Beams& beams) {
    Kokkos::parallel_for(
        "CalculateConstraintX0", solver.num_constraint_nodes,
        CalculateConstraintX0{solver.constraints.node_indices, beams.node_x0, solver.constraints.X0}
    );
}

void UpdateStatePrediction(Solver& solver, View_N x_system, View_N x_lambda) {
    // Update state prediction based on system solution
    if (solver.is_dynamic_solve) {
        // Calculate change in state based on dynamic solution iteration
        Kokkos::parallel_for(
            "UpdateDynamicPrediction", solver.num_system_nodes,
            UpdateDynamicPrediction{
                solver.h, solver.beta_prime, solver.gamma_prime, x_system, solver.state.q_delta,
                solver.state.v, solver.state.vd}
        );
    } else {
        Kokkos::parallel_for(
            // Calculate change in state based on static solution iteration
            "UpdateStaticPrediction", solver.num_system_nodes,
            UpdateStaticPrediction{
                solver.h, solver.beta_prime, solver.gamma_prime, x_system, solver.state.q_delta}
        );
    }

    // Update predicted displacements
    Kokkos::parallel_for(
        "CalculateDisplacement", solver.num_system_nodes,
        CalculateDisplacement{solver.h, solver.state.q_delta, solver.state.q_prev, solver.state.q}
    );

    // If constraints are being used, update state lambda
    if (solver.num_constraint_nodes > 0) {
        // Update lambda in state
        Kokkos::parallel_for(
            "UpdateLambdaPrediction", solver.num_constraint_dofs,
            UpdateLambdaPrediction{x_lambda, solver.state.lambda}
        );
    }
}

template <typename Subview_NxN, typename Subview_N>
void AssembleSystem(Solver& solver, Beams& beams, Subview_NxN St_11, Subview_N R_system) {
    // Tangent operator
    Kokkos::deep_copy(solver.T, 0.);
    Kokkos::parallel_for(
        "TangentOperator", solver.num_system_nodes,
        CalculateTangentOperator{solver.h, solver.state.q_delta, solver.T}
    );

    // Assemble residual vector
    Kokkos::deep_copy(R_system, 0.);
    AssembleResidualVector(beams, R_system);

    // Assemble matrices from beam elements
    Kokkos::deep_copy(solver.K, 0.);
    AssembleElasticStiffnessMatrix(beams, solver.K);
    AssembleInertialStiffnessMatrix(beams, solver.K);

    // Iteration matrix
    Kokkos::deep_copy(St_11, 0.);
    KokkosBlas::gemm("N", "N", 1.0, solver.K, solver.T, 1.0, St_11);

    // If dynamic solution
    if (solver.is_dynamic_solve) {
        Kokkos::deep_copy(solver.M, 0.);
        Kokkos::deep_copy(solver.G, 0.);
        AssembleMassMatrix(beams, solver.M);
        AssembleGyroscopicInertiaMatrix(beams, solver.G);
        KokkosBlas::axpy(solver.beta_prime, solver.M, St_11);
        KokkosBlas::axpy(solver.gamma_prime, solver.G, St_11);
    }
}

template <typename Subview_NxN, typename Subview_N>
void AssembleConstraints(
    Solver& solver, Subview_NxN St_12, Subview_NxN St_21, Subview_N R_system, Subview_N R_lambda
) {
    // If no constraints in solver, return
    if (solver.num_constraint_dofs == 0) {
        return;
    }

    // Constraint residual vector and gradient matrix
    Kokkos::deep_copy(solver.constraints.Phi, 0.);
    Kokkos::deep_copy(solver.constraints.B, 0.);
    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", solver.num_constraint_nodes,
        CalculateConstraintResidualGradient{
            solver.constraints.node_indices, solver.constraints.X0, solver.constraints.u,
            solver.state.q, solver.constraints.Phi, solver.constraints.B}
    );

    // Update residual vector
    KokkosBlas::gemv("T", 1.0, solver.constraints.B, solver.state.lambda, 1.0, R_system);
    Kokkos::deep_copy(R_lambda, solver.constraints.Phi);

    // Update iteration matrix
    Kokkos::parallel_for(
        "St_12=B.transpose", 1,
        KOKKOS_LAMBDA(size_t) {
            for (size_t i = 0; i < solver.num_constraint_dofs; ++i) {
                for (size_t j = 0; j < solver.num_system_dofs; ++j) {
                St_12(j, i) = solver.constraints.B(i, j);
            }
            }
        }
    );
    KokkosBlas::gemm("N", "N", 1.0, solver.constraints.B, solver.T, 0.0, St_21);
}

void solve_linear_system(View_NxN system, View_N solution) {
    auto A =
        Kokkos::View<double**, Kokkos::LayoutLeft>("system", system.extent(0), system.extent(1));
    Kokkos::deep_copy(A, system);
    auto b = Kokkos::View<double*, Kokkos::LayoutLeft>("solution", solution.extent(0));
    Kokkos::deep_copy(b, solution);
    auto pivots =
        Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::HostSpace>("pivots", solution.extent(0));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
    KokkosBlas::gesv(A, b, pivots);
#pragma GCC diagnostic pop

    // Kokkos::deep_copy(system, A);
    Kokkos::deep_copy(solution, b);
}

void SolveSystem(Solver& solver) {
    // Condition system for solve
    Kokkos::parallel_for(
        "ConditionSystem", 1,
        ConditionSystem{
            solver.num_system_dofs, solver.num_dofs, solver.conditioner, solver.St, solver.R}
    );

    // Solve linear system
    KokkosBlas::axpby(-1.0, solver.R, 0.0, solver.x);
    // openturbine::gebt_poc::solve_linear_system(solver.St, solver.x);
    solve_linear_system(solver.St, solver.x);

    // Uncondition solution
    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs,
        UnconditionSolution{solver.num_system_dofs, solver.conditioner, solver.x}
    );
}

double CalculateConvergenceError(Solver& solver) {
    const double atol = 1e-7;
    const double rtol = 1e-5;
    double sum_error_squared = 0.;
    Kokkos::parallel_reduce(
        solver.num_system_dofs, CalculateErrorSumSquares{atol, rtol, solver.state.q_delta, solver.x},
        sum_error_squared
    );
    solver.convergence_err = std::sqrt(sum_error_squared / solver.num_dofs);
    return solver.convergence_err;
}

bool Step(Solver& solver, Beams& beams) {
    // Predict state at end of step
    PredictNextState(solver);

    auto system_range = Kokkos::make_pair((size_t)0, solver.num_system_dofs);
    auto constraint_range = Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs);

    auto R_system = Kokkos::subview(solver.R, system_range);
    auto R_lambda = Kokkos::subview(solver.R, constraint_range);

    auto x_system = Kokkos::subview(solver.x, system_range);
    auto x_lambda = Kokkos::subview(solver.x, constraint_range);

    auto St_11 = Kokkos::subview(solver.St, system_range, system_range);
    auto St_12 = Kokkos::subview(solver.St, system_range, constraint_range);
    auto St_21 = Kokkos::subview(solver.St, constraint_range, system_range);

    // Perform convergence iterations
    for (size_t iter = 0; iter < solver.max_iter; ++iter) {
        Kokkos::deep_copy(solver.St, 0.);
        Kokkos::deep_copy(solver.R, 0.);

        // Update beam elements state from solvers
        UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);

        // Assemble iteration matrix and residual vector
        AssembleSystem(solver, beams, St_11, R_system);

        // Assemble constraints
        AssembleConstraints(solver, St_12, St_21, R_system, R_lambda);

        // Solve system
        SolveSystem(solver);

        // Calculate error
        auto err = CalculateConvergenceError(solver);

        // If error is sufficiently small, solution converged, update acceleration and return
        if (err < 1.) {
            Kokkos::parallel_for(
                "UpdateAlgorithmicAcceleration", solver.num_system_nodes,
                KOKKOS_LAMBDA(size_t i) {
                    for (size_t j = 0; j < kLieAlgebraComponents; ++j) {
                        solver.state.a(i, j) +=
                            (1. - solver.alpha_f) / (1 - solver.alpha_m) * solver.state.vd(i, j);
                    }
                }
            );

            // Solution converged
            return true;
        }

        // Update state prediction
        UpdateStatePrediction(solver, x_system, x_lambda);
    }

    // Solution did not converge
    return false;
}

}  // namespace openturbine