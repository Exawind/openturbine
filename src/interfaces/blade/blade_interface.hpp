#pragma once

#include <filesystem>

#include "interfaces/components/beam.hpp"
#include "interfaces/components/beam_input.hpp"
#include "interfaces/components/solution_input.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/outputs.hpp"
#include "model/model.hpp"
#include "state/clone_state.hpp"
#include "state/copy_state_data.hpp"
#include "step/step.hpp"
#include "utilities/netcdf/node_state_writer.hpp"

namespace openturbine::interfaces {

/**
 * @brief Interface for blade simulation that manages state, solver, and components
 *
 * This class represents the primary interface for simulating a WT blade, connecting
 * the blade components with the solver and state management.
 */
class BladeInterface {
public:
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

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
          state(model.CreateState<DeviceType>()),
          elements(model.CreateElements<DeviceType>()),
          constraints(model.CreateConstraints<DeviceType>()),
          parameters(
              solution_input.dynamic_solve, solution_input.max_iter, solution_input.time_step,
              solution_input.rho_inf, solution_input.absolute_error_tolerance,
              solution_input.relative_error_tolerance
          ),
          solver(CreateSolver<DeviceType>(state, elements, constraints)),
          state_save(CloneState(state)),
          host_state(state) {
        // Update the host state with current node motion
        this->host_state.CopyFromState(this->state);

        // Update the blade motion from state
        this->blade.GetMotion(this->host_state);

        // Initialize NetCDF writer and write mesh connectivity if output path is specified
        if (!solution_input.output_file_path.empty()) {
            // Create output directory if it doesn't exist
            std::filesystem::create_directories(solution_input.output_file_path);

            // Write mesh connectivity to YAML file
            model.ExportMeshConnectivityToYAML(
                solution_input.output_file_path + "/mesh_connectivity.yaml"
            );

            // Initialize outputs
            this->outputs = std::make_unique<Outputs>(
                solution_input.output_file_path + "/blade_interface.nc", blade.nodes.size()
            );
        }
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
            for (auto i = 0U; i < 6; ++i) {
                this->host_state.f(node.id, i) = node.loads[i];
            }
        }
        Kokkos::deep_copy(this->state.f, this->host_state.f);

        // Solve for state at end of step
        auto converged = openturbine::Step(
            this->parameters, this->solver, this->elements, this->state, this->constraints
        );
        if (!converged) {
            return false;
        }

        // Update the host state with current node motion
        this->host_state.CopyFromState(this->state);

        // Update the blade motion from state
        this->blade.GetMotion(this->host_state);

        // Write outputs and increment timestep counter
        if (this->outputs) {
            outputs->WriteNodeOutputsAtTimestep(this->host_state, this->state.time_step);
        }

        return true;
    }

    /// @brief Saves the current state for potential restoration (in correction step)
    void SaveState() { CopyStateData(this->state_save, this->state); }

    /// @brief Restores the previously saved state (in correction step)
    void RestoreState() {
        CopyStateData(this->state, this->state_save);

        // Update the host state with current node motion
        this->host_state.CopyFromState(this->state);

        // Update the blade motion from state
        this->blade.GetMotion(this->host_state);
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

private:
    Model model;              ///< OpenTurbine class for model construction
    components::Beam blade;   ///< Blade model input/output data
    State<DeviceType> state;  ///< OpenTurbine class for storing system state
    Elements<DeviceType>
        elements;  ///< OpenTurbine class for model elements (beams, masses, springs)
    Constraints<DeviceType>
        constraints;               ///< OpenTurbine class for constraints tying elements together
    StepParameters parameters;     ///< OpenTurbine class containing solution parameters
    Solver<DeviceType> solver;     ///< OpenTurbine class for solving the dynamic system
    State<DeviceType> state_save;  ///< OpenTurbine class state class for temporarily saving state
    HostState<DeviceType> host_state;  ///< Host local copy of node state data
    std::unique_ptr<Outputs> outputs;  ///< handle to Output for writing to NetCDF
};

}  // namespace openturbine::interfaces
