#pragma once

#include <filesystem>

#include "interfaces/components/controller_input.hpp"
#include "interfaces/components/solution_input.hpp"
#include "interfaces/components/turbine.hpp"
#include "interfaces/components/turbine_input.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/outputs.hpp"
#include "model/model.hpp"
#include "state/clone_state.hpp"
#include "state/copy_state_data.hpp"
#include "step/step.hpp"
#include "utilities/controllers/turbine_controller.hpp"

namespace openturbine::interfaces {

/**
 * @brief Interface for blade simulation that manages state, solver, and components
 *
 * This class represents the primary interface for simulating a WT blade, connecting
 * the blade components with the solver and state management.
 */
class TurbineInterface {
public:
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

    /**
     * @brief Constructs a TurbineInterface from solution and blade inputs
     * @param solution_input Configuration parameters for solver and solution
     * @param blade_input Configuration parameters for blade geometry
     */
    explicit TurbineInterface(
        const components::SolutionInput& solution_input,
        const components::TurbineInput& turbine_input,
        const components::ControllerInput& controller_input = {}
    )
        : model(Model(solution_input.gravity)),
          turbine(turbine_input, model),
          state(model.CreateState<DeviceType>()),
          elements(model.CreateElements<DeviceType>()),
          constraints(model.CreateConstraints<DeviceType>()),
          parameters(
              solution_input.dynamic_solve, solution_input.max_iter, solution_input.time_step,
              solution_input.rho_inf, solution_input.absolute_error_tolerance,
              solution_input.relative_error_tolerance
          ),
          solver(CreateSolver(state, elements, constraints)),
          state_save(CloneState(state)),
          host_state(state) {
        // Initialize controller if library path is provided
        if (controller_input.IsEnabled()) {
            try {
                controller = std::make_unique<util::TurbineController>(
                    controller_input.shared_lib_path, controller_input.function_name,
                    controller_input.input_file_path, controller_input.simulation_name
                );

                // Initialize controller with turbine and solution parameters
                InitializeController(turbine_input, solution_input);
            } catch (const std::runtime_error& e) {
                std::cerr << "Warning: Failed to load controller library '"
                          << controller_input.shared_lib_path << "': " << e.what() << std::endl;
                std::cerr << "Continuing without controller." << std::endl;
            }
        }

        // Update the host state with current node motion
        this->host_state.CopyFromState(this->state);

        // Update the turbine node motion based on the host state
        this->turbine.GetMotion(this->host_state);

        // Initialize NetCDF writer and write mesh connectivity if output path is specified
        if (!solution_input.output_file_path.empty()) {
            // Create output directory if it doesn't exist
            std::filesystem::create_directories(solution_input.output_file_path);

            // Write mesh connectivity to YAML file
            model.ExportMeshConnectivityToYAML(
                solution_input.output_file_path + "/mesh_connectivity.yaml"
            );

            // Initialize outputs with both node state and time-series files
            this->outputs = std::make_unique<Outputs>(
                solution_input.output_file_path + "/turbine_interface.nc",
                solution_input.output_file_path + "/turbine_time_series.nc",
                this->state.num_system_nodes
            );

            // Write initial state
            this->outputs->WriteNodeOutputsAtTimestep(this->host_state, this->state.time_step);

            // Write initial time-series data (test values)
            this->outputs->WriteRotorTimeSeriesAtTimestep(
                this->state.time_step, turbine_input.azimuth_angle, turbine_input.rotor_speed
            );
        }
    }

    /// @brief Returns a reference to the turbine model
    [[nodiscard]] components::Turbine& Turbine() { return this->turbine; }

    /**
     * @brief Steps forward in time
     * @return true if solver converged, false otherwise
     * @note This function updates the host state with current node loads,
     *       solves the dynamic system, and updates the node motion with the new state.
     *       If the solver does not converge, the motion is not updated.
     */
    [[nodiscard]] bool Step() {
        // Update the host state with current node loads
        this->turbine.SetLoads(this->host_state);
        Kokkos::deep_copy(this->state.f, this->host_state.f);

        // Solve for state at end of step
        auto converged = openturbine::Step(
            this->parameters, this->solver, this->elements, this->state, this->constraints
        );

        // If not converged, return false
        if (!converged) {
            return false;
        }

        // Update the host state with current node motion
        this->host_state.CopyFromState(this->state);

        // Update the turbine node motion based on the host state
        this->turbine.GetMotion(this->host_state);

        // Write outputs and increment timestep counter
        if (this->outputs) {
            // Write node state outputs
            this->outputs->WriteNodeOutputsAtTimestep(this->host_state, this->state.time_step);

            // Calculate rotor azimuth and speed -> write rotor time-series data
            this->WriteRotorTimeSeriesData();
        }

        return true;
    }

    /// @brief Saves the current state for potential restoration (in correction step)
    void SaveState() { CopyStateData(this->state_save, this->state); }

    /// @brief Restores the previously saved state (in correction step)
    void RestoreState() {
        // Copy saved state back to current state
        CopyStateData(this->state, this->state_save);

        // Update the host state with current node motion
        this->host_state.CopyFromState(this->state);

        // Update the turbine node motion based on the host state
        this->turbine.GetMotion(this->host_state);
    }

private:
    Model model;                  ///< OpenTurbine class for model construction
    components::Turbine turbine;  ///< Turbine model input/output data
    State<DeviceType> state;      ///< OpenTurbine class for storing system state
    Elements<DeviceType>
        elements;  ///< OpenTurbine class for model elements (beams, masses, springs)
    Constraints<DeviceType>
        constraints;               ///< OpenTurbine class for constraints tying elements together
    StepParameters parameters;     ///< OpenTurbine class containing solution parameters
    Solver<DeviceType> solver;     ///< OpenTurbine class for solving the dynamic system
    State<DeviceType> state_save;  ///< OpenTurbine class state class for temporarily saving state
    HostState<DeviceType> host_state;                     ///< Host local copy of node state data
    std::unique_ptr<Outputs> outputs;                     ///< handle to Output for writing to NetCDF
    std::unique_ptr<util::TurbineController> controller;  ///< DISCON-style controller

    /**
     * @brief Write rotor time-series data based on constraint outputs
     *
     * This method extracts rotor azimuth angle and speed from the constraint system
     * and writes them to the time-series output file. Data is read from the shaft
     * base to azimuth constraint, which contains:
     * - Index 0: Azimuth angle (radians)
     * - Index 1: Rotor speed (rad/s)
     */
    void WriteRotorTimeSeriesData() {
        if (!this->outputs) {
            return;
        }

        // Write the calculated values to time-series output
        this->outputs->WriteRotorTimeSeriesAtTimestep(
            this->state.time_step,
            this->CalculateAzimuthAngle(),  // azimuth angle (radians)
            this->CalculateRotorSpeed()     // rotor speed (rad/s)
        );
    }

    /**
     * @brief Calculates and normalizes azimuth angle from constraint output
     * @return Azimuth angle in radians, normalized to [0, 2π)
     */
    [[nodiscard]] double CalculateAzimuthAngle() const {
        const auto azimuth_constraint_id = this->turbine.shaft_base_to_azimuth.id;
        double azimuth = this->constraints.host_output(azimuth_constraint_id, 0);

        // Normalize azimuth angle to range [0, 2π) radians
        azimuth = std::fmod(azimuth, 2. * M_PI);
        if (azimuth < 0) {
            azimuth += 2. * M_PI;
        }

        return azimuth;
    }

    /**
     * @brief Calculates rotor speed from constraint output
     * @return Rotor speed in rad/s
     */
    [[nodiscard]] double CalculateRotorSpeed() const {
        const auto azimuth_constraint_id = this->turbine.shaft_base_to_azimuth.id;
        return this->constraints.host_output(azimuth_constraint_id, 1);
    }

    /**
     * @brief Initialize controller with turbine parameters and connect to constraints
     * @param turbine_input Configuration parameters for turbine geometry and initial conditions
     */
    void InitializeController(
        const components::TurbineInput& turbine_input,
        const components::SolutionInput& solution_input
    ) {
        if (!controller) {
            return;
        }

        // Set controller constant parameters
        controller->io.dt = solution_input.time_step;           // Time step size (seconds)
        controller->io.pitch_actuator_type_req = 0;             // Pitch position actuator
        controller->io.pitch_control_type = 0;                  // Collective pitch control
        controller->io.n_blades = turbine_input.blades.size();  // Number of blades

        // Set controller initial values
        controller->io.time = 0.;                                    // Current time (seconds)
        controller->io.azimuth_angle = turbine_input.azimuth_angle;  // Initial azimuth
        controller->io.pitch_blade1_actual = turbine_input.blade_pitch_angle;  // Blade pitch (rad)
        controller->io.pitch_blade2_actual = turbine_input.blade_pitch_angle;  // Blade pitch (rad)
        controller->io.pitch_blade3_actual = turbine_input.blade_pitch_angle;  // Blade pitch (rad)
        controller->io.generator_speed_actual =
            turbine_input.rotor_speed * turbine_input.gear_box_ratio;  // Generator speed (rad/s)
        controller->io.generator_torque_actual =
            turbine_input.generator_power /
            (turbine_input.rotor_speed * turbine_input.gear_box_ratio);  // Generator torque
        controller->io.generator_power_actual =
            turbine_input.generator_power;                                    // Generator power (W)
        controller->io.rotor_speed_actual = turbine_input.rotor_speed;        // Rotor speed (rad/s)
        controller->io.horizontal_wind_speed = turbine_input.hub_wind_speed;  // Hub wind speed (m/s)

        // Signal first call to controller
        controller->io.status = 0;

        // Make first call to controller to initialize
        controller->CallController();
    }
};

}  // namespace openturbine::interfaces
