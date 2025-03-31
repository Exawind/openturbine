#pragma once

#include "interfaces/components/blade.hpp"
#include "interfaces/components/blade_input.hpp"
#include "interfaces/components/solution_input.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/vtk_output.hpp"
#include "model/model.hpp"
#include "state/clone_state.hpp"
#include "state/copy_state_data.hpp"
#include "step/step.hpp"

namespace openturbine::interfaces {

class BladeInterface {
public:
    explicit BladeInterface(
        const components::SolutionInput& solution_input, const components::BladeInput& blade_input
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

    /// @brief Step forward in time
    [[nodiscard]] bool Step() {
        // Transfer node loads to state
        for (const auto& node : this->blade.nodes) {
            for (auto j = 0U; j < 6; ++j) {
                state.host_f(node.id, j) = node.loads[j];
            }
        }

        // Solve for state at end of step
        auto converged = openturbine::Step(
            this->parameters, this->solver, this->elements, this->state, this->constraints
        );

        // If not converged, return false
        if (!converged) {
            return false;
        }

        // Update the blade motion
        UpdateNodeMotion();

        return true;
    }

    /// @brief Save state for correction step
    void SaveState() {
        // Copy current state data into saved state
        CopyStateData(this->state_save, this->state);
    }

    /// @brief Restore state for correction step
    void RestoreState() {
        // Restore current state from saved state
        CopyStateData(this->state, this->state_save);

        // Update the turbine motion to match restored state
        UpdateNodeMotion();
    }

    /// @brief Set root node displacement if `prescribe_root_motion` input was true
    void SetRootDisplacement(const std::array<double, 7>& u) const {
        if (this->blade.prescribed_root_constraint_id == kInvalidID) {
            throw("prescribed root motion was not enabled");
        }
        this->constraints.UpdateDisplacement(this->blade.prescribed_root_constraint_id, u);
    }

    void WriteOutputVTK() {
        this->vtk_output.WriteBeam(this->blade.nodes);
        this->vtk_output.IncrementFileIndex();
    }

private:
    /// @brief  OpenTurbine class used for model construction
    Model model;

public:
    /// @brief Blade model input/output data
    components::Blade blade;

private:
    /// @brief  OpenTurbine class for storing system state
    State state;

    /// @brief  OpenTurbine class for model elements (beams, masses, springs)
    Elements elements;

    /// @brief  OpenTurbine class for constraints tying elements together
    Constraints constraints;

    /// @brief  OpenTurbine class containing solution parameters
    StepParameters parameters;

    /// @brief  OpenTurbine class for solving the dynamic system
    Solver solver;

    /// @brief  OpenTurbine class state class for temporarily saving state
    State state_save;

    /// @brief  Host local copy of node position, displacement, velocity, acceleration
    HostState host_state;

    /// @brief  VTK output manager
    VTKOutput vtk_output;

    /// @brief  Update motion data for all nodes in the interface
    void UpdateNodeMotion() {
        // Copy state motion members from device to host
        this->host_state.CopyFromState(this->state);

        // Set blade root node motion
        this->blade.root_node.UpdateMotion(this->host_state);

        // Loop through blade nodes and set node motion
        for (auto& node : this->blade.nodes) {
            node.UpdateMotion(this->host_state);
        }
    }
};

}  // namespace openturbine::interfaces
