#pragma once

#include <filesystem>

#include "interfaces/components/blade.hpp"
#include "interfaces/components/blade_input.hpp"
#include "interfaces/components/solution_input.hpp"
#include "interfaces/host_state.hpp"
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
    /**
     * @brief Constructs a BladeInterface from solution and blade inputs
     * @param solution_input Configuration parameters for solver and solution
     * @param blade_input Configuration parameters for blade geometry
     */
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
          host_state(state) {
        // Update the blade motion to match state
        UpdateNodeMotion();

        // Initialize NetCDF writer and write mesh connectivity if output path is specified
        if (!solution_input.output_file_path.empty()) {
            // Create output directory if it doesn't exist
            std::filesystem::create_directories(solution_input.output_file_path);

            // Initialize NetCDF writer
            this->output_writer_ = std::make_unique<util::NodeStateWriter>(
                solution_input.output_file_path + "/blade_interface.nc", true, blade.nodes.size()
            );

            // Write mesh connectivity to YAML file
            model.ExportMeshConnectivityToYAML(
                solution_input.output_file_path + "/mesh_connectivity.yaml"
            );
        }
    }

    /// @brief Returns a reference to the blade model
    [[nodiscard]] components::Blade& Blade() { return this->blade; }

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
        if (!converged) {
            return false;
        }

        // Update the blade motion if there was convergence
        UpdateNodeMotion();

        // Write outputs and increment timestep counter
        if (this->output_writer_) {
            WriteOutputs();
        }
        this->current_timestep_++;

        return true;
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

private:
    Model model;                  ///< OpenTurbine class for model construction
    components::Blade blade;      ///< Blade model input/output data
    State state;                  ///< OpenTurbine class for storing system state
    Elements elements;            ///< OpenTurbine class for model elements (beams, masses, springs)
    Constraints constraints;      ///< OpenTurbine class for constraints tying elements together
    StepParameters parameters;    ///< OpenTurbine class containing solution parameters
    Solver solver;                ///< OpenTurbine class for solving the dynamic system
    State state_save;             ///< OpenTurbine class state class for temporarily saving state
    HostState host_state;         ///< Host local copy of node state data
    size_t current_timestep_{0};  ///< Current timestep index
    std::unique_ptr<util::NodeStateWriter> output_writer_;  ///< NetCDF output writer

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

    //------------------------------------------------------------------------------
    // Write outputs
    //------------------------------------------------------------------------------
    void WriteOutputs() const {
        const size_t num_blade_nodes = this->blade.nodes.size();
        std::vector<double> x(num_blade_nodes);
        std::vector<double> y(num_blade_nodes);
        std::vector<double> z(num_blade_nodes);
        std::vector<double> i(num_blade_nodes);
        std::vector<double> j(num_blade_nodes);
        std::vector<double> k(num_blade_nodes);
        std::vector<double> w(num_blade_nodes);

        // write position
        for (size_t idx = 0; idx < num_blade_nodes; ++idx) {
            const auto& node = this->blade.nodes[idx];
            x[idx] = this->host_state.x(node.id, 0);
            y[idx] = this->host_state.x(node.id, 1);
            z[idx] = this->host_state.x(node.id, 2);
            i[idx] = this->host_state.x(node.id, 3);
            j[idx] = this->host_state.x(node.id, 4);
            k[idx] = this->host_state.x(node.id, 5);
            w[idx] = this->host_state.x(node.id, 6);
        }
        output_writer_->WriteStateDataAtTimestep(current_timestep_, "x", x, y, z, i, j, k, w);

        // write displacement
        for (size_t idx = 0; idx < num_blade_nodes; ++idx) {
            const auto& node = this->blade.nodes[idx];
            x[idx] = this->host_state.q(node.id, 0);
            y[idx] = this->host_state.q(node.id, 1);
            z[idx] = this->host_state.q(node.id, 2);
            w[idx] = this->host_state.q(node.id, 3);
            i[idx] = this->host_state.q(node.id, 4);
            j[idx] = this->host_state.q(node.id, 5);
            k[idx] = this->host_state.q(node.id, 6);
        }
        output_writer_->WriteStateDataAtTimestep(current_timestep_, "u", x, y, z, i, j, k, w);

        // write velocity
        for (size_t idx = 0; idx < num_blade_nodes; ++idx) {
            const auto& node = this->blade.nodes[idx];
            x[idx] = this->host_state.v(node.id, 0);
            y[idx] = this->host_state.v(node.id, 1);
            z[idx] = this->host_state.v(node.id, 2);
            i[idx] = this->host_state.v(node.id, 3);
            j[idx] = this->host_state.v(node.id, 4);
            k[idx] = this->host_state.v(node.id, 5);
        }
        output_writer_->WriteStateDataAtTimestep(current_timestep_, "v", x, y, z, i, j, k);

        // write acceleration
        for (size_t idx = 0; idx < num_blade_nodes; ++idx) {
            const auto& node = this->blade.nodes[idx];
            x[idx] = this->host_state.vd(node.id, 0);
            y[idx] = this->host_state.vd(node.id, 1);
            z[idx] = this->host_state.vd(node.id, 2);
            i[idx] = this->host_state.vd(node.id, 3);
            j[idx] = this->host_state.vd(node.id, 4);
            k[idx] = this->host_state.vd(node.id, 5);
        }
        output_writer_->WriteStateDataAtTimestep(current_timestep_, "a", x, y, z, i, j, k);
    }
};

}  // namespace openturbine::interfaces
