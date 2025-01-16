#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_constraints_matrix.hpp"
#include "assemble_constraints_residual.hpp"
#include "assemble_system_matrix.hpp"
#include "assemble_system_residual.hpp"
#include "assemble_tangent_operator.hpp"
#include "calculate_convergence_error.hpp"
#include "predict_next_state.hpp"
#include "reset_constraints.hpp"
#include "solve_system.hpp"
#include "step_parameters.hpp"
#include "update_constraint_prediction.hpp"
#include "update_constraint_variables.hpp"
#include "update_state_prediction.hpp"
#include "update_system_variables.hpp"
#include "update_tangent_operator.hpp"

#include "src/constraints/calculate_constraint_output.hpp"
#include "src/constraints/constraints.hpp"
#include "src/elements/elements.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/state/update_algorithmic_acceleration.hpp"
#include "src/state/update_global_position.hpp"

namespace openturbine {

/**
 * @brief Attempts to complete a single time step in the dynamic FEA simulation
 *
 * @param parameters Simulation step parameters including time step size and convergence criteria
 * @param solver     Solver object containing system matrices and solution methods
 * @param elements   Collection of elements (beams, masses etc.) in the FE mesh
 * @param state      Current state of the system (positions, velocities, accelerations etc.)
 * @param constraints System constraints and their associated data
 *
 * @return true if the step converged within the maximum allowed iterations, otherwise false
 */
inline bool Step(
    StepParameters& parameters, Solver& solver, Elements& elements, State& state,
    Constraints& constraints
) {
    auto region = Kokkos::Profiling::ScopedRegion("Step");

    PredictNextState(parameters, state);
    ResetConstraints(constraints);

    solver.convergence_err.clear();
    double err{1000.};
    for (auto iter = 0U; err > 1.; ++iter) {
        if (iter >= parameters.max_iter) {
            return false;
        }

        UpdateSystemVariables(parameters, elements, state);

        UpdateTangentOperator(parameters, state);

        AssembleTangentOperator(solver, state);

        AssembleSystemResidual(solver, elements, state);

        AssembleSystemMatrix(solver, elements);

        UpdateConstraintVariables(state, constraints);

        AssembleConstraintsMatrix(solver, constraints);

        AssembleConstraintsResidual(solver, constraints);

        SolveSystem(parameters, solver);

        err = CalculateConvergenceError(parameters, solver, state, constraints);

        solver.convergence_err.push_back(err);

        UpdateStatePrediction(parameters, solver, state);

        UpdateConstraintPrediction(solver, constraints);
    }

    Kokkos::parallel_for(
        "UpdateAlgorithmicAcceleration", solver.num_system_nodes,
        UpdateAlgorithmicAcceleration{
            state.a,
            state.vd,
            parameters.alpha_f,
            parameters.alpha_m,
        }
    );

    Kokkos::parallel_for(
        "UpdateGlobalPosition", solver.num_system_nodes,
        UpdateGlobalPosition{
            state.q,
            state.x0,
            state.x,
        }
    );

    Kokkos::parallel_for(
        "CalculateConstraintOutput", constraints.num_constraints,
        CalculateConstraintOutput{
            constraints.type,
            constraints.target_node_index,
            constraints.axes,
            state.x0,
            state.q,
            state.v,
            state.vd,
            constraints.output,
        }
    );
    Kokkos::deep_copy(constraints.host_output, constraints.output);

    return true;
}

}  // namespace openturbine
