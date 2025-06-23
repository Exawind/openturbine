#pragma once

namespace openturbine::interfaces::components {

/**
 * @brief Configuration parameters for a DISCON-style turbine controller
 *
 * This struct encapsulates all the necessary configuration parameters needed to
 * initialize and configure a DISCON-style wind turbine controller. DISCON is a
 * standardized interface for wind turbine control algorithms that allows for
 * dynamic loading of controller implementations at runtime.
 */
struct ControllerInput {
    std::string shared_lib_path;  ///< Path to controller shared library
    std::string function_name;    ///< Controller function name (default: "DISCON")
    std::string input_file_path;  ///< Path to controller input file
    std::string simulation_name;  ///< Simulation name for controller

    /// @brief Default constructor - creates empty controller input
    ControllerInput() = default;

    /**
     * @brief Constructor with all parameters
     * @param shared_lib_path Path to the shared library containing the controller implementation
     * @param function_name Name of the controller function to call (defaults to "DISCON")
     * @param input_file_path Optional path to controller-specific configuration file
     * @param simulation_name Optional identifier for the simulation run
     */
    ControllerInput(
        const std::string& shared_lib_path, const std::string& function_name = "DISCON",
        const std::string& input_file_path = "", const std::string& simulation_name = ""
    )
        : shared_lib_path(shared_lib_path),
          function_name(function_name),
          input_file_path(input_file_path),
          simulation_name(simulation_name) {}

    /// @brief Check if controller is enabled (i.e. has library path)
    [[nodiscard]] bool IsEnabled() const { return !shared_lib_path.empty(); }
};

}  // namespace openturbine::interfaces::components
