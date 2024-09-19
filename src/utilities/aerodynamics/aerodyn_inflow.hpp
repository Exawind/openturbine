#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::util {

/// Struct for error handling settings
struct ErrorHandling {
    /// Error levels used in InflowWind
    enum class ErrorLevel {
        kNone = 0,
        kInfo = 1,
        kWarning = 2,
        kSevereError = 3,
        kFatalError = 4
    };

    static constexpr size_t kErrorMessagesLength = 1025U;  //< Max error message length in Fortran
    int abort_error_level{
        static_cast<int>(ErrorLevel::kFatalError)};          //< Error level at which to abort
    int error_status{0};                                     //< Error status
    std::array<char, kErrorMessagesLength> error_message{};  //< Error message buffer

    /// Check for errors and throw an exception if found
    bool CheckError() const {
        return error_status == 0 ? true : throw std::runtime_error(error_message.data());
    }
};

/// Struct to hold the properties of the working fluid (air)
struct FluidProperties {
    double density{1.225};                 // Air density (kg/m^3)
    double kinematic_viscosity{1.464E-5};  // Kinematic viscosity (m^2/s)
    double sound_speed{335.};              // Speed of sound in the working fluid (m/s)
    double vapor_pressure{1700.};          // Vapor pressure of the working fluid (Pa)
};

/// Struct to hold the environmental conditions
struct EnvironmentalConditions {
    double gravity{9.80665};       // Gravitational acceleration (m/s^2)
    double atm_pressure{103500.};  // Atmospheric pressure (Pa)
    double water_depth{0.};        // Water depth (m)
    double msl_offset{0.};         // Mean sea level to still water level offset (m)
};

/// Struct to hold the settings for the turbine (assuming a single turbine)
struct TurbineSettings {
    int n_turbines{1};                                      // Number of turbines - 1 by default
    int n_blades{3};                                        // Number of blades - 3 by default
    std::array<float, 3> initial_hub_position{0.};          // Initial hub position
    std::array<double, 9> initial_hub_orientation{0.};      // Initial hub orientation
    std::array<float, 3> initial_nacelle_position{0.};      // Initial nacelle position
    std::array<double, 9> initial_nacelle_orientation{0.};  // Initial nacelle orientation
    std::array<float, 3> initial_root_position{0.};         // Initial root position
    std::array<double, 9> initial_root_orientation{0.};     // Initial root orientation
};

/// Struct to hold the structural mesh data
struct StructuralMesh {
    int n_mesh_points{1};                                       // Number of mesh points
    std::vector<std::array<float, 3>> initial_mesh_position{};  // N x 3 array [x, y, z]
    std::vector<std::array<double, 9>>
        initial_mesh_orientation{};  // N x 9 array [r11, r12, ..., r33]
    std::vector<int>
        mesh_point_to_blade_num{};  // N x 1 array for mapping a mesh point to blade number
};

/// Struct to hold the settings for the simulation controls
struct SimulationControls {
    static constexpr size_t kDefaultStringLength{1025};  // Max length for output filenames

    // Input file handling
    bool aerodyn_input_passed{true};     // Input file passed for AeroDyn
    bool inflowwind_input_passed{true};  // Input file passed for InflowWind

    // Interpolation order (must be either 1: linear, or 2: quadratic)
    int interpolation_order{1};  // Interpolation order - linear by default

    // Initial time related variables
    float time_step{0.1f};          // Simulation timestep (s)
    float max_time{600.f};          // Maximum simulation time (s)
    float total_elapsed_time{0.f};  // Total elapsed time (s)
    int num_time_steps{0};          // Number of time steps

    // Flags
    int store_HH_wind_speed{1};  // Store hub-height wind speed?
    int transpose_DCM{1};        // Transpose the direction cosine matrix?
    int debug_level{0};          // Debug level (0-4)

    // Outputs
    int output_format{0};        // File format for writing outputs
    float output_time_step{0.};  // Timestep for outputs to file
    std::array<char, kDefaultStringLength> output_root_name{
        "Output_ADIlib_default"  // Root name for output files
    };
    int n_channels{0};                              // Number of channels returned
    std::array<char, 20 * 8000> channel_names_c{};  // Output channel names
    std::array<char, 20 * 8000> channel_units_c{};  // Output channel units
};

/// Struct to hold the settings for VTK output
struct VTKSettings {
    bool write_vtk{false};                      // Flag to write VTK output
    int vtk_type{1};                            // Type of VTK output (1: surface meshes)
    std::array<float, 6> vtk_nacelle_dimensions{// Nacelle dimensions for VTK rendering
                                                -2.5f, -2.5f, 0.f, 10.f, 5.f, 5.f};
    float vtk_hub_radius{1.5f};  // Hub radius for VTK rendering
};

/// Wrapper class for the AeroDynInflow (ADI) shared library
/// @details The AeroDynInflow (ADI) shared library is a Fortran library that provides C bindings
/// for interfacing with the AeroDyn Inflow module
struct AeroDynInflowLibrary {
    util::dylib lib{
        "libaerodyn_inflow_c_binding.dylib",
        util::dylib::no_filename_decorations};  //< Dynamic library object for AeroDyn Inflow
    ErrorHandling error_handling;               //< Error handling settings
    FluidProperties air;                        //< Properties of the working fluid (air)
    EnvironmentalConditions env_conditions;     //< Environmental conditions
    TurbineSettings turbine_settings;           //< Turbine settings
    StructuralMesh structural_mesh;             //< Structural mesh data
    SimulationControls sim_controls;            //< Simulation control settings
    VTKSettings vtk_settings;                   //< VTK output settings

    AeroDynInflowLibrary(std::string shared_lib_path = "") {
        if (!shared_lib_path.empty()) {
            lib = util::dylib(shared_lib_path, util::dylib::no_filename_decorations);
        }
    }

    /// Wrapper for ADI_C_PreInit routine to initialize AeroDyn Inflow library
    void ADI_PreInit() {
        auto ADI_C_PreInit =
            this->lib.get_function<void(int*, int*, int*, int*, char*)>("ADI_C_PreInit");
        ADI_C_PreInit(
            &turbine_settings.n_turbines,        // input: Number of turbines
            &sim_controls.transpose_DCM,         // input: Transpose DCM?
            &sim_controls.debug_level,           // input: Debug level
            &error_handling.error_status,        // output: Error status
            error_handling.error_message.data()  // output: Error message
        );

        error_handling.CheckError();
    }

    /// Wrapper for ADI_C_SetupRotor routine to set up the rotor
    void ADI_SetupRotor(
        int turbine_number, int is_horizontal_axis, std::vector<float> turbine_ref_pos
    ) {
        auto ADI_C_SetupRotor = this->lib.get_function<
            void(int*, int*, float*, float*, double*, float*, double*, int*, float*, double*, int*, float*, double*, int*, int*, char*)>(
            "ADI_C_SetupRotor"
        );

        // Flatten mesh arrays
        auto init_mesh_pos_flat = FlattenArray(structural_mesh.initial_mesh_position);
        auto init_mesh_orient_flat = FlattenArray(structural_mesh.initial_mesh_orientation);

        ADI_C_SetupRotor(
            &turbine_number,                               // input: current turbine number
            &is_horizontal_axis,                           // input: 1: HAWT, 0: VAWT or cross-flow
            turbine_ref_pos.data(),                        // input: turbine reference position
            turbine_settings.initial_hub_position.data(),  // input: initial hub position
            turbine_settings.initial_hub_orientation.data(),   // input: initial hub orientation
            turbine_settings.initial_nacelle_position.data(),  // input: initial nacelle position
            turbine_settings.initial_nacelle_orientation.data(
            ),                                              // input: initial nacelle orientation
            &turbine_settings.n_blades,                     // input: number of blades
            turbine_settings.initial_root_position.data(),  // input: initial blade root positions
            turbine_settings.initial_root_orientation.data(
            ),                               // input: initial blade root orientation
            &structural_mesh.n_mesh_points,  // input: number of mesh points
            init_mesh_pos_flat.data(),       // input: initial node positions
            init_mesh_orient_flat.data(),    // input: initial node orientation
            structural_mesh.mesh_point_to_blade_num.data(
            ),                                   // input: initial mesh point to blade number mapping
            &error_handling.error_status,        // output: Error status
            error_handling.error_message.data()  // output: Error message buffer
        );

        error_handling.CheckError();
    }

private:
    /// Method to flatten a 2D array into a 1D array for Fortran compatibility
    template <typename T, size_t N>
    std::vector<T> FlattenArray(const std::vector<std::array<T, N>>& input) {
        std::vector<T> output;
        for (const auto& arr : input) {
            output.insert(output.end(), arr.begin(), arr.end());
        }
        return output;
    }
};

}  // namespace openturbine::util
