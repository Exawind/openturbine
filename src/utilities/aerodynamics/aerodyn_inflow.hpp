#pragma once

#include <stdexcept>

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

    static constexpr size_t kErrorMessagesLength{1025U};     // Max error message length in Fortran
    int abort_error_level{4};                                // Error level at which to abort
    int error_status{0};                                     // Error status
    std::array<char, kErrorMessagesLength> error_message{};  // Error message buffer
};

/// Struct to hold the environmental conditions related to the working fluid i.e. air
struct EnvironmentalConditions {
    double gravity{9.80665};                        // Gravitational acceleration (m/s^2)
    double density{1.225};                          // Air density (kg/m^3)
    double kinematic_viscosity{1.464E-05};          // Kinematic viscosity (m^2/s)
    double sound_speed{335.};                       // Speed of sound in working fluid (m/s)
    double atm_pressure{103500.};                   // Atmospheric pressure (Pa)
    double vapor_pressure{1700.};                   // Vapour pressure of working fluid (Pa)
    double water_depth{0.};                         // Water depth (m)
    double mean_sea_level_2_still_water_level{0.};  // Offset (m)
};

/// Struct to hold the settings for the turbine (assuming a single turbine)
struct TurbineSettings {
    int n_turbines{1};                                  // Number of turbines - 1 by default
    int n_blades{3};                                    // Number of blades - 3 by default
    std::array<int, 3> initial_hub_position{0, 0, 0};   // Initial hub position
    std::array<int, 9> initial_hub_orientation{0};      // Initial hub orientation
    std::array<int, 3> initial_nacelle_position{0};     // Initial nacelle position
    std::array<int, 9> initial_nacelle_orientation{0};  // Initial nacelle orientation
    std::array<int, 3> initial_root_position{0};        // Initial root position
    std::array<int, 9> initial_root_orientation{0};     // Initial root orientation
};

/// Struct to hold the structural mesh data
struct StructuralMesh {
    int n_mesh_points{1};                                       // Number of mesh points
    std::vector<std::array<float, 3>> initial_mesh_position{};  // N x 3 array [x, y, z]
    std::vector<std::array<double, 9>>
        initial_mesh_orientation{};              // N x 9 array [r11, r12, ..., r33]
    std::vector<int> mesh_point_to_blade_num{};  // N x 1 array [blade number]
};

/// Struct to hold the settings for the simulation controls
struct SimulationControls {
    static constexpr size_t kDefaultStringLength{
        1025};  // Max length of the name used for any output file written by the HD Fortran code

    // Input file handling
    bool aerodyn_input_passed{true};     // Assume passing of input file as a string
    bool inflowwind_input_passed{true};  // Assume passing of input file as a string

    // Interpolation order (must be 1: linear, or 2: quadratic)
    int interpolation_order{1};  // Interpolation order - linear by default

    // Initial time related variables
    float dt{0.1f};         // Timestep (s)
    float tmax{600.f};      // Maximum time (s)
    float total_time{0.f};  // Total elapsed time (s)
    int n_time_steps{0};    // Number of time steps

    // Flags
    int store_HH_wind_speed{1};  // Store hub-height wind speed?
    int transpose_DCM{1};        // Transpose the direction cosine matrix?
    int debug_level{0};          // Debug level (0-4)

    // Outputs
    int output_format{0};       // File format for writing outputs
    float output_timestep{0.};  // Timestep for outputs to file
    std::array<char, kDefaultStringLength> output_root_name{
        "Output_ADIlib_default"  // Root name for output files
    };
    int n_channels{0};                              // Number of channels returned
    std::array<char, 20 * 8000> channel_names_c{};  // Output channel names
    std::array<char, 20 * 8000> channel_units_c{};  // Output channel units
};

/// Struct to hold the settings for writing VTK output
struct VTKSettings {
    bool write_vtk{false};  // Write VTK output?
    int vtk_type{1};        // VTK output type (1: surface meshes)
    std::array<float, 6> vtk_nacelle_dimensions{
        -2.5f, -2.5f, 0.f,
        10.f,  5.f,   5.f};  // Nacelle dimensions for VTK surface rendering [x0,y0,z0,Lx,Ly,Lz] (m)
    float VTKHubRad{1.5f};   // Hub radius for VTK surface rendering
};

/// @brief Wrapper class for the AeroDynInflow (ADI) shared library
struct AeroDynInflowLibrary {
    std::string library_path;                //< Path to the shared library
    ErrorHandling error_handling;            //< Error handling settings
    EnvironmentalConditions env_conditions;  //< Environmental conditions
    TurbineSettings turbine_settings;        //< Turbine settings
    SimulationControls sim_controls;         //< Simulation controls
    VTKSettings vtk_settings;                //< VTK settings

    AeroDynInflowLibrary(const std::string& path) : library_path(path) {}
};

}  // namespace openturbine::util
