#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_constraints_matrix.hpp"
#include "assemble_constraints_residual.hpp"
#include "assemble_system_matrix.hpp"
#include "assemble_system_residual.hpp"
#include "calculate_convergence_error.hpp"
#include "constraints/calculate_constraint_output.hpp"
#include "constraints/constraints.hpp"
#include "elements/elements.hpp"
#include "predict_next_state.hpp"
#include "reset_constraints.hpp"
#include "reset_solver.hpp"
#include "solve_system.hpp"
#include "solver/solver.hpp"
#include "state/state.hpp"
#include "state/update_algorithmic_acceleration.hpp"
#include "state/update_global_position.hpp"
#include "step_parameters.hpp"
#include "update_constraint_prediction.hpp"
#include "update_constraint_variables.hpp"
#include "update_state_prediction.hpp"
#include "update_system_variables.hpp"
#include "update_tangent_operator.hpp"

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
template <typename DeviceType>
inline bool Step(
    StepParameters& parameters, Solver<DeviceType>& solver, Elements<DeviceType>& elements,
    State<DeviceType>& state, Constraints<DeviceType>& constraints
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
        ResetSolver(solver);

        UpdateTangentOperator(parameters, state);

        UpdateSystemVariables(parameters, elements, state);

        AssembleSystemResidual(solver, elements, state);

        AssembleSystemMatrix(parameters, solver, elements);

        UpdateConstraintVariables(state, constraints);

        AssembleConstraintsMatrix(solver, constraints);

        AssembleConstraintsResidual(solver, constraints);

        SolveSystem(parameters, solver);

        err = CalculateConvergenceError(parameters, solver, state, constraints);

        solver.convergence_err.push_back(err);

        UpdateStatePrediction(parameters, solver, state);

        UpdateConstraintPrediction(solver, constraints);
    }

    auto system_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, solver.num_system_nodes);
    Kokkos::parallel_for(
        "UpdateAlgorithmicAcceleration", system_range,
        UpdateAlgorithmicAcceleration<DeviceType>{
            state.a,
            state.vd,
            parameters.alpha_f,
            parameters.alpha_m,
        }
    );

    Kokkos::parallel_for(
        "UpdateGlobalPosition", system_range,
        UpdateGlobalPosition<DeviceType>{
            state.q,
            state.x0,
            state.x,
        }
    );

    auto constraints_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, constraints.num_constraints);
    Kokkos::parallel_for(
        "CalculateConstraintOutput", constraints_range,
        CalculateConstraintOutput<DeviceType>{
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
