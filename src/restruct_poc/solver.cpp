#include "solver.hpp"

#include <KokkosLapack_gesv.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "beams.hpp"
#include "CalculateNextState.hpp"
#include "CalculateDisplacement.hpp"
#include "CalculateConstraintX0.hpp"
#include "UpdateDynamicPrediction.hpp"
#include "UpdateStaticPrediction.hpp"
#include "UpdateLambdaPrediction.hpp"
#include "ConditionSystem.hpp"
#include "CalculateErrorSumSquares.hpp"
#include "UpdateAlgorithmicAcceleration.hpp"
#include "CalculateTangentOperator.hpp"
#include "CalculateConstraintResidualGradient.hpp"
#include "UpdateIterationMatrix.hpp"

namespace openturbine {

void PredictNextState(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Predict Next State");
    Kokkos::deep_copy(solver.state.lambda, 0.);
    Kokkos::deep_copy(solver.state.q_prev, solver.state.q);

    // Predict the new state values
    Kokkos::parallel_for(
        "CalculateNextState", solver.num_system_nodes,
        CalculateNextState{
            solver.h,
            solver.alpha_f,
            solver.alpha_m,
            solver.beta,
            solver.gamma,
            solver.state.q_delta,
            solver.state.v,
            solver.state.vd,
            solver.state.a,
        }
    );

    // Update predicted displacements
    Kokkos::parallel_for(
        "CalculateDisplacement", solver.state.q.extent(0),
        CalculateDisplacement{
            solver.h,
            solver.state.q_delta,
            solver.state.q_prev,
            solver.state.q,
        }
    );
}

void InitializeConstraints(Solver& solver, Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Initialize Constraints");
    Kokkos::parallel_for(
        "CalculateConstraintX0", solver.num_constraint_nodes,
        CalculateConstraintX0{
            solver.constraints.node_indices,
            beams.node_x0,
            solver.constraints.X0,
        }
    );
}

void UpdateStatePrediction(Solver& solver, View_N x_system, View_N x_lambda) {
    auto region = Kokkos::Profiling::ScopedRegion("Update State Prediction");
    // Update state prediction based on system solution
    if (solver.is_dynamic_solve) {
        // Calculate change in state based on dynamic solution iteration
        Kokkos::parallel_for(
            "UpdateDynamicPrediction", solver.num_system_nodes,
            UpdateDynamicPrediction{
                solver.h,
                solver.beta_prime,
                solver.gamma_prime,
                x_system,
                solver.state.q_delta,
                solver.state.v,
                solver.state.vd,
            }
        );
    } else {
        Kokkos::parallel_for(
            // Calculate change in state based on static solution iteration
            "UpdateStaticPrediction", solver.num_system_nodes,
            UpdateStaticPrediction{
                solver.h,
                solver.beta_prime,
                solver.gamma_prime,
                x_system,
                solver.state.q_delta,
            }
        );
    }

    // Update predicted displacements
    Kokkos::parallel_for(
        "CalculateDisplacement", solver.num_system_nodes,
        CalculateDisplacement{
            solver.h,
            solver.state.q_delta,
            solver.state.q_prev,
            solver.state.q,
        }
    );

    // If constraints are being used, update state lambda
    if (solver.num_constraint_nodes > 0) {
        // Update lambda in state
        Kokkos::parallel_for(
            "UpdateLambdaPrediction", solver.num_constraint_dofs,
            UpdateLambdaPrediction{
                x_lambda,
                solver.state.lambda,
            }
        );
    }
}

template <typename Subview_NxN, typename Subview_N>
void AssembleSystem(Solver& solver, Beams& beams, Subview_NxN St_11, Subview_N R_system) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System");
    // Tangent operator
    Kokkos::deep_copy(solver.T, 0.);
    Kokkos::parallel_for(
        "TangentOperator", solver.num_system_nodes,
        CalculateTangentOperator{
            solver.h,
            solver.state.q_delta,
            solver.T,
        }
    );

    // Assemble residual vector
    Kokkos::deep_copy(R_system, 0.);
    AssembleResidualVector(beams, R_system);

    // Assemble matrices from beam elements
    Kokkos::deep_copy(solver.K, 0.);
    AssembleElasticStiffnessMatrix(beams, solver.K);
    AssembleInertialStiffnessMatrix(beams, solver.K);

    // Iteration matrix
    KokkosBlas::gemm("N", "N", 1.0, solver.K, solver.T, 0.0, St_11);

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
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Constraints");
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
            solver.constraints.node_indices,
            solver.constraints.X0,
            solver.constraints.u,
            solver.state.q,
            solver.constraints.Phi,
            solver.constraints.B,
        }
    );

    // Update residual vector
    KokkosBlas::gemv("T", 1.0, solver.constraints.B, solver.state.lambda, 1.0, R_system);
    Kokkos::deep_copy(R_lambda, solver.constraints.Phi);

    // Update iteration matrix
    Kokkos::parallel_for(
        "St_12=B.transpose",
        Kokkos::MDRangePolicy{{0, 0}, {solver.num_constraint_dofs, solver.num_system_dofs}},
        UpdateIterationMatrix<Subview_NxN>{St_12, solver.constraints.B}
    );

    KokkosBlas::gemm("N", "N", 1.0, solver.constraints.B, solver.T, 0.0, St_21);
}

void SolveSystem(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");
    // Condition system for solve
    Kokkos::parallel_for(
        "PreconditionSt",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {solver.num_system_dofs, solver.num_dofs}),
        PreconditionSt{
            solver.St,
            solver.conditioner,
        }
    );
    Kokkos::parallel_for(
        "PostconditionSt",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, solver.num_system_dofs}, {solver.num_dofs, solver.num_dofs}
        ),
        PostconditionSt{
            solver.St,
            solver.conditioner,
        }
    );
    Kokkos::parallel_for(
        "ConditionR", solver.num_system_dofs,
        ConditionR{
            solver.R,
            solver.conditioner,
        }
    );

    // Solve linear system
    KokkosBlas::axpby(-1.0, solver.R, 0.0, solver.x);
    Kokkos::deep_copy(solver.St_left, solver.St);
    auto x = Kokkos::View<double*, Kokkos::LayoutLeft>(solver.x);
    KokkosLapack::gesv(solver.St_left, x, solver.IPIV);
    Kokkos::deep_copy(solver.x, x);

    // Uncondition solution
    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs,
        UnconditionSolution{
            solver.num_system_dofs,
            solver.conditioner,
            solver.x,
        }
    );
}

double CalculateConvergenceError(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Calculate Convergence Error");
    const double atol = 1e-7;
    const double rtol = 1e-5;
    double sum_error_squared = 0.;
    Kokkos::parallel_reduce(
        solver.num_system_dofs,
        CalculateErrorSumSquares{
            atol,
            rtol,
            solver.state.q_delta,
            solver.x,
        },
        sum_error_squared
    );
    return std::sqrt(sum_error_squared / solver.num_system_dofs);
}

bool Step(Solver& solver, Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Step");
    // Predict state at end of step
    PredictNextState(solver);

    auto system_range = Kokkos::make_pair(0, solver.num_system_dofs);
    auto constraint_range = Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs);

    auto R_system = Kokkos::subview(solver.R, system_range);
    auto R_lambda = Kokkos::subview(solver.R, constraint_range);

    auto x_system = Kokkos::subview(solver.x, system_range);
    auto x_lambda = Kokkos::subview(solver.x, constraint_range);

    auto St_11 = Kokkos::subview(solver.St, system_range, system_range);
    auto St_12 = Kokkos::subview(solver.St, system_range, constraint_range);
    auto St_21 = Kokkos::subview(solver.St, constraint_range, system_range);

    // Reset convergence error vector
    solver.convergence_err.clear();

    // Initialize convergence error to a large number
    double err = 1000.0;

    // Loop while error is greater than 1
    for (int iter = 0; err > 1.0; ++iter) {
        // Initialize iteration matrix and residual to zero
        Kokkos::deep_copy(solver.St, 0.);

        // Update beam elements state from solvers
        UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);

        // Assemble iteration matrix and residual vector
        AssembleSystem(solver, beams, St_11, R_system);

        // Assemble constraints
        AssembleConstraints(solver, St_12, St_21, R_system, R_lambda);

        // Solve system
        SolveSystem(solver);

        // Calculate error for this iteration
        err = CalculateConvergenceError(solver);

        // Save convergence error in vector
        solver.convergence_err.push_back(err);

        // Update state prediction
        UpdateStatePrediction(solver, x_system, x_lambda);

        // If iteration reaches maximum, return solution failed to converge
        if (iter >= solver.max_iter) {
            return false;
        }
    }

    // Solution converged, update algorithmic acceleration for next step
    Kokkos::parallel_for(
        "UpdateAlgorithmicAcceleration", solver.num_system_nodes,
        UpdateAlgorithmicAcceleration{
            solver.state.a,
            solver.state.vd,
            solver.alpha_f,
            solver.alpha_m,
        }
    );

    // Return solution converged
    return true;
}

}  // namespace openturbine
