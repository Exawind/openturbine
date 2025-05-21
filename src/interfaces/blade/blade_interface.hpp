#pragma once

#include "interfaces/components/beam.hpp"
#include "interfaces/components/beam_input.hpp"
#include "interfaces/components/solution_input.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/vtk_output.hpp"
#include "model/model.hpp"
#include "state/clone_state.hpp"
#include "state/copy_state_data.hpp"
#include "step/step.hpp"

namespace openturbine::interfaces {

/**
 * @brief Interface for blade simulation that manages state, solver, and components
 *
 * This class represents the primary interface for simulating a WT blade, connecting
 * the blade components with the solver and state management.
 */
class BladeInterface {
public:
    /**
     * @brief Constructs a BladeInterface from solution and blade inputs
     * @param solution_input Configuration parameters for solver and solution
     * @param blade_input Configuration parameters for blade geometry
     */
    explicit BladeInterface(
        const components::SolutionInput& solution_input, const components::BeamInput& blade_input
    )
        : model(Model(solution_input.gravity)),
          blade(blade_input, model),
          state(model.CreateState()),
          elements(model.CreateElements()),
          constraints(model.CreateConstraints()),
          parameters(
              solution_input.dynamic_solve, solution_input.max_iter, solution_input.time_step,
              solution_input.rho_inf, solution_input.absolute_error_tolerance,
              solution_input.relative_error_tolerance
          ),
          solver(CreateSolver(state, elements, constraints)),
          state_save(CloneState(state)),
          host_state(state),
          vtk_output(solution_input.vtk_output_path) {
        // Update the blade motion to match state
        UpdateNodeMotion();
    }

    /// @brief Returns a reference to the blade model
    [[nodiscard]] components::Beam& Blade() { return this->blade; }

    /**
     * @brief Steps forward in time
     * @return true if solver converged, false otherwise
     */
    [[nodiscard]] bool Step() {
        // Transfer node loads -> state
        for (const auto& node : this->blade.nodes) {
            for (auto j = 0U; j < 6; ++j) {
                state.host_f(node.id, j) = node.loads[j];
            }
        }

        // Solve for state at end of step
        auto converged = openturbine::Step(
            this->parameters, this->solver, this->elements, this->state, this->constraints
        );

        // Update the blade motion if there was convergence
        return converged ? (UpdateNodeMotion(), true) : false;
    }

    /// @brief Saves the current state for potential restoration (in correction step)
    void SaveState() { CopyStateData(this->state_save, this->state); }

    /// @brief Restores the previously saved state (in correction step)
    void RestoreState() {
        CopyStateData(this->state, this->state_save);
        UpdateNodeMotion();
    }

    /**
     * @brief Sets the displacement for the root node
     * @param u Displacement array (7 components)
     * @throws std::runtime_error if prescribed root motion was not enabled
     */
    void SetRootDisplacement(const std::array<double, 7>& u) const {
        if (this->blade.prescribed_root_constraint_id == kInvalidID) {
            throw std::runtime_error("prescribed root motion was not enabled");
        }
        this->constraints.UpdateDisplacement(this->blade.prescribed_root_constraint_id, u);
    }

    ///@brief Writes the current blade state to VTK output files
    void WriteOutputVTK() {
        this->vtk_output.WriteBeam(this->blade.nodes);
        this->vtk_output.IncrementFileIndex();
    }

private:
    Model model;                ///< OpenTurbine class for model construction
    components::Beam blade;     ///< Blade model input/output data
    State state;                ///< OpenTurbine class for storing system state
    Elements elements;          ///< OpenTurbine class for model elements (beams, masses, springs)
    Constraints constraints;    ///< OpenTurbine class for constraints tying elements together
    StepParameters parameters;  ///< OpenTurbine class containing solution parameters
    Solver solver;              ///< OpenTurbine class for solving the dynamic system
    State state_save;           ///< OpenTurbine class state class for temporarily saving state
    HostState host_state;       ///< Host local copy of node state data
    VTKOutput vtk_output;       ///< VTK output manager

    /// @brief  Updates motion data for all nodes (root and blade) in the interface
    void UpdateNodeMotion() {
        // Copy state motion members from device to host
        this->host_state.CopyFromState(this->state);

        // Update all node motion
        this->blade.root_node.UpdateMotion(this->host_state);
        for (auto& node : this->blade.nodes) {
            node.UpdateMotion(this->host_state);
        }
    }
};

}  // namespace openturbine::interfaces
