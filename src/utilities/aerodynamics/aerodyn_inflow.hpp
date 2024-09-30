#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/math/quaternion_operations.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::util {

/**
 * AeroDynInflowLibrary: A C++ wrapper for the AeroDyn/InflowWind (ADI) shared library
 *
 * ## Overview
 *
 * This wrapper simplifies interaction with the ADI library, providing a modern C++ interface for
 * OpenTurbine developers to utilize AeroDyn and InflowWind functionality. It encapsulates the
 * C-bindings of the Fortran library, drawing inspiration from the existing Python
 * interface.
 *
 * ## Features
 *
 * AeroDyn utilizes blade element momentum (BEM) theory to calculate aerodynamic forces acting
 * on each blade section. It accounts for factors such as:
 *  - Dynamic stall (Beddoes-Leishman or OLAF models)
 *  - Unsteady aerodynamics
 *  - Tower shadow effects
 *  - Wind shear
 *  - Tip and hub losses
 *
 * InflowWind simulates the inflow conditions around wind turbines by modeling spatially and
 * temporally varying wind fields. It enables the simulation of complex wind phenomena, including:
 *  - Atmospheric turbulence (e.g., Kaimal, von Karman spectra)
 *  - Wind shear (power law or logarithmic profiles)
 *  - Discrete gusts and extreme events
 *  - Various wind field types (uniform, full-field turbulence, user-defined)
 *
 * ## Usage
 *
 * 1. Instantiate the AeroDynInflowLibrary class with the path to the shared library
 * 2. Initialize the AeroDyn/InflowWind modules:
 *    - PreInitialize(): Set up general parameters
 *    - SetupRotor(): Configure rotor-specific settings (iterate over turbines)
 *    - Initialize(): Complete initialization with input files
 * 3. Perform the simulation by iterating over timesteps:
 *    - SetupRotorMotion(): Update rotor motion (iterate over turbines)
 *    - UpdateStates(): Advance internal states
 *    - CalculateOutputChannels(): Compute output values
 *    - GetRotorAerodynamicLoads(): Retrieve aerodynamic forces and moments
 * 4. Complete the simulation:
 *    - Finalize(): Clean up and release resources
 *    - Handle any resulting errors using the ErrorHandling struct
 *
 * Note: Refer to the unit tests for more detailed examples of how to use this wrapper.
 *
 * ## References
 *
 * - OpenFAST/AeroDyn documentation:
 *   https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html
 * - OpenFAST/InflowWind documentation:
 *   https://openfast.readthedocs.io/en/main/source/user/inflowwind/index.html
 * - AeroDyn InflowWind C bindings:
 *   https://github.com/OpenFAST/openfast/blob/dev/modules/aerodyn/src/AeroDyn_Inflow_C_Binding.f90
 */

/**
 * @brief Converts a 7-element array of position and quaternion to separate position and orientation
 * arrays
 * @param data Input array: [x, y, z, qw, qx, qy, qz]
 * @param position Output array for position [x, y, z]
 * @param orientation Output array for flattened 3x3 rotation matrix
 */
inline void SetPositionAndOrientation(
    const std::array<double, 7>& data, std::array<float, 3>& position,
    std::array<double, 9>& orientation
) {
    // Set position (first 3 elements)
    for (size_t i = 0; i < 3; ++i) {
        position[i] = static_cast<float>(data[i]);
    }

    // Set orientation (convert last 4 elements to 3x3 rotation matrix)
    auto orientation_2D = QuaternionToRotationMatrix({data[3], data[4], data[5], data[6]});

    // Flatten the 3x3 matrix to a 1D array
    std::copy(&orientation_2D[0][0], &orientation_2D[0][0] + 9, orientation.begin());
}

/**
 * @brief Struct to hold the motion data of any structural mesh component
 *
 * @details This struct holds the motion data (i.e. position, orientation,
 * velocity, and acceleration) of the structural mesh, which can be the hub, nacelle, root, or
 * mesh points/nodes.
 */
struct MeshData {
    int n_mesh_points;
    std::vector<std::array<float, 3>> position;
    std::vector<std::array<double, 9>> orientation;
    std::vector<std::array<float, 6>> velocity;
    std::vector<std::array<float, 6>> acceleration;
    std::vector<std::array<float, 6>> loads;

    /// Default constructor
    MeshData() = default;

    /// Constructor to initialize all data based on provided inputs
    MeshData(size_t n_nodes)
        : n_mesh_points(static_cast<int>(n_nodes)),
          position(std::vector<std::array<float, 3>>(n_nodes, {0., 0., 0.})),
          orientation(
              std::vector<std::array<double, 9>>(n_nodes, {0., 0., 0., 0., 0., 0., 0., 0., 0.})
          ),
          velocity(std::vector<std::array<float, 6>>(n_nodes, {0., 0., 0., 0., 0., 0.})),
          acceleration(std::vector<std::array<float, 6>>(n_nodes, {0., 0., 0., 0., 0., 0.})),
          loads(std::vector<std::array<float, 6>>(n_nodes, {0., 0., 0., 0., 0., 0.})) {}
};

struct TurbineConfig {
    struct BladeConfig {
        Array_7 root_initial_pos;
        std::vector<Array_7> node_initial_pos;
    };
    std::array<float, 3> reference_pos{0., 0., 0.};
    Array_7 hub_initial_pos;
    Array_7 nacelle_initial_pos;
    std::vector<BladeConfig> blades;
    bool is_horizontal_axis{true};
};

struct TurbineData {
    int32_t n_blades;
    MeshData hub;
    MeshData nacelle;
    MeshData blade_roots;
    MeshData blade_nodes;
    std::vector<int32_t> blade_nodes_to_blade_num;
    std::vector<std::vector<size_t>> blade_node_index;

    TurbineData(const TurbineConfig& tc)
        : n_blades(static_cast<int32_t>(tc.blades.size())),
          hub(1),
          nacelle(1),
          blade_roots(tc.blades.size()),
          blade_nodes(std::accumulate(
              tc.blades.begin(), tc.blades.end(), 0U,
              [](size_t sum, const TurbineConfig::BladeConfig& bc) {
                  return sum + bc.node_initial_pos.size();
              }
          )),
          blade_node_index(std::vector<std::vector<size_t>>(tc.blades.size(), std::vector<size_t>{})
          ) {
        SetPositionAndOrientation(tc.hub_initial_pos, hub.position[0], hub.orientation[0]);
        SetPositionAndOrientation(
            tc.nacelle_initial_pos, nacelle.position[0], nacelle.orientation[0]
        );

        size_t i_blade{0};
        size_t i_node{0};
        for (const auto& bc : tc.blades) {
            SetPositionAndOrientation(
                bc.root_initial_pos, blade_roots.position[i_blade], blade_roots.orientation[i_blade]
            );
            for (const auto& bn : bc.node_initial_pos) {
                SetPositionAndOrientation(
                    bn, blade_nodes.position[i_node], blade_nodes.orientation[i_node]
                );
                blade_node_index.back().emplace_back(i_node);
                blade_nodes_to_blade_num.emplace_back(static_cast<int>(i_blade + 1));
                ++i_node;
            }
            ++i_blade;
        }
    }
};

/**
 * @brief Struct for error handling settings
 *
 * @details This struct holds the error handling settings for the AeroDynInflow library wrapper. It
 * includes an error level enum, a maximum error message length, and methods for checking and
 * handling errors.
 */
struct ErrorHandling {
    /// Error levels used in InflowWind
    enum class ErrorLevel {
        kNone = 0,
        kInfo = 1,
        kWarning = 2,
        kSevereError = 3,
        kFatalError = 4
    };

    static constexpr size_t kErrorMessagesLength{1025U};  //< Max error message length
    int abort_error_level{
        static_cast<int>(ErrorLevel::kFatalError)  //< Error level at which to abort
    };
    int error_status{0};                                     //< Error status
    std::array<char, kErrorMessagesLength> error_message{};  //< Error message buffer

    /// Checks for errors and throws an exception if found
    void CheckError() const {
        if (error_status != 0) {
            throw std::runtime_error(error_message.data());
        }
    }
};

/// Struct to hold the properties of the working fluid (air)
struct FluidProperties {
    float density{1.225f};                 //< Air density (kg/m^3)
    float kinematic_viscosity{1.464E-5f};  //< Kinematic viscosity (m^2/s)
    float sound_speed{335.f};              //< Speed of sound in the working fluid (m/s)
    float vapor_pressure{1700.f};          //< Vapor pressure of the working fluid (Pa)
};

/// Struct to hold the environmental conditions
struct EnvironmentalConditions {
    float gravity{9.80665f};       //< Gravitational acceleration (m/s^2)
    float atm_pressure{103500.f};  //< Atmospheric pressure (Pa)
    float water_depth{0.f};        //< Water depth (m)
    float msl_offset{0.f};         //< Mean sea level to still water level offset (m)
};

/**
 * @brief Struct to hold the settings for simulation controls
 *
 * @details This struct holds the settings for simulation controls, including input file
 * handling, interpolation order, time-related variables, and flags.
 */
struct SimulationControls {
    static constexpr size_t kDefaultStringLength{1025};  //< Max length for output filenames

    // Input file handling
    std::string aerodyn_input;
    bool aerodyn_input_is_path{true};
    std::string inflowwind_input;
    bool inflowwind_input_is_path{true};

    // Interpolation order (must be either 1: linear, or 2: quadratic)
    int interpolation_order{1};  //< Interpolation order - linear by default

    // Initial time related variables
    double time_step{0.1};          //< Simulation timestep (s)
    double max_time{600.};          //< Maximum simulation time (s)
    double total_elapsed_time{0.};  //< Total elapsed time (s)
    int n_time_steps{0};            //< Number of time steps

    // Flags
    int store_HH_wind_speed{1};  //< Flag to store HH wind speed

    // Outputs
    int output_format{1};                                //< File format for writing outputs
    double output_time_step{0.1};                        //< Timestep for outputs to file
    std::array<char, 1025> output_root_name{"ADI_out"};  //< Root name for output files
    int n_channels{0};                                   //< Number of channels returned
    std::array<char, 20 * 8000> channel_names_c{};       //< Output channel names
    std::array<char, 20 * 8000> channel_units_c{};       //< Output channel units
};

/**
 * @brief Struct to hold the settings for VTK output
 *
 * @details This struct holds the settings for VTK output, including the flag to write VTK
 * output, the type of VTK output, and the nacelle dimensions for VTK rendering.
 */
struct VTKSettings {
    int write_vtk{false};                       //< Flag to write VTK output
    int vtk_type{1};                            //< Type of VTK output (1: surface meshes)
    std::array<float, 6> vtk_nacelle_dimensions{//< Nacelle dimensions for VTK rendering
                                                -2.5f, -2.5f, 0.f, 10.f, 5.f, 5.f};
    float vtk_hub_radius{1.5f};  //< Hub radius for VTK rendering
};

/**
 * @brief Wrapper class for the AeroDynInflow (ADI) shared library
 *
 * @details This class provides an interface for interacting with the AeroDynInflow (ADI) shared
 * library, which is a Fortran library offering C bindings for the AeroDyn x InflowWind modules
 * of OpenFAST.
 *
 * The class encapsulates the following key functions:
 *  - PreInitialize (ADI_C_PreInit): Pre-initializes the AeroDynInflow module
 *  - SetupRotor (ADI_C_SetupRotor): Configures rotor-specific parameters before simulation
 *  - Initialize (ADI_C_Init): Initializes the AeroDynInflow module for simulation
 *  - SetupRotorMotion (ADI_C_SetRotorMotion): Sets rotor motion for a given time step
 *  - GetRotorAerodynamicLoads (ADI_C_GetRotorLoads): Retrieves aerodynamic loads on the rotor
 *  - CalculateOutputChannels (ADI_C_CalcOutput): Calculates output channels at a given time
 *  - UpdateStates (ADI_C_UpdateStates): Updates states for the next time step
 *  - Finalize (ADI_C_End): Ends the AeroDynInflow module and frees memory
 */
struct AeroDynInflowLibrary {
    enum class DebugLevel {
        None = 0,
        Basic = 1,
        Low = 2,
        Medium = 3,
        Max = 4,
    };

    util::dylib lib{
        "libaerodyn_inflow_c_binding.dylib",
        util::dylib::no_filename_decorations  //< Dynamic library object for AeroDyn Inflow
    };
    ErrorHandling error_handling;            //< Error handling settings
    FluidProperties air;                     //< Properties of the working fluid (air)
    EnvironmentalConditions env_conditions;  //< Environmental conditions
    SimulationControls sim_controls;         //< Simulation control settings
    VTKSettings vtk_settings;                //< VTK output settings

    std::vector<TurbineData> turbines;

    /// Constructor to initialize AeroDyn Inflow library with default settings and optional path
    AeroDynInflowLibrary(
        std::string shared_lib_path = "", SimulationControls sc = SimulationControls{},
        VTKSettings vtk = VTKSettings{}
    )
        : sim_controls(sc), vtk_settings(vtk) {
        if (!shared_lib_path.empty()) {
            lib = util::dylib(shared_lib_path, util::dylib::no_filename_decorations);
        }
    }

    /// Wrapper for ADI_C_PreInit routine to initialize AeroDyn Inflow library
    void Initialize(
        std::vector<TurbineConfig> turbine_configs, DebugLevel debug_level = DebugLevel::None,
        bool transpose_dcm = false
    ) {
        //----------------------------------------------------------------------
        // ADI_C_PreInit
        //----------------------------------------------------------------------

        auto ADI_C_PreInit =
            this->lib.get_function<void(int*, int*, int*, int*, char*)>("ADI_C_PreInit");

        int32_t debug_level_int{static_cast<int>(debug_level)};
        int32_t transpose_dcm_int{transpose_dcm ? 1 : 0};
        int32_t n_turbines{static_cast<int32_t>(turbine_configs.size())};

        ADI_C_PreInit(
            &n_turbines,                         // input: Number of turbines
            &transpose_dcm_int,                  // input: Transpose DCM?
            &debug_level_int,                    // input: Debug level
            &error_handling.error_status,        // output: Error status
            error_handling.error_message.data()  // output: Error message
        );

        error_handling.CheckError();

        //----------------------------------------------------------------------
        // ADI_C_SetupRotor
        //----------------------------------------------------------------------

        auto ADI_C_SetupRotor = this->lib.get_function<
            void(int*, int*, const float*, float*, double*, float*, double*, int*, float*, double*, int*, float*, double*, int*, int*, char*)>(
            "ADI_C_SetupRotor"
        );

        int32_t turbine_number{0};

        // Loop through turbine configurations
        for (const auto& tc : turbine_configs) {
            // Turbine number (1 is first turbine)
            ++turbine_number;

            // Get is HAWT as integer
            int32_t is_horizontal_axis{tc.is_horizontal_axis ? 1 : 0};

            // Create new turbine data
            turbines.emplace_back(TurbineData(tc));

            // Get data
            auto& td = turbines.back();

            // Call setup rotor for each turbine
            ADI_C_SetupRotor(
                &turbine_number,                            // input: current turbine number
                &is_horizontal_axis,                        // input: 1: HAWT, 0: VAWT or cross-flow
                tc.reference_pos.data(),                    // input: turbine reference position
                td.hub.position.data()->data(),             // input: initial hub position
                td.hub.orientation.data()->data(),          // input: initial hub orientation
                td.nacelle.position.data()->data(),         // input: initial nacelle position
                td.nacelle.orientation.data()->data(),      // input: initial nacelle orientation
                &td.n_blades,                               // input: number of blades
                td.blade_roots.position.data()->data(),     // input: initial blade root positions
                td.blade_roots.orientation.data()->data(),  // input: initial blade root orientation
                &td.blade_nodes.n_mesh_points,              // input: number of mesh points
                td.blade_nodes.position.data()->data(),     // input: initial node positions
                td.blade_nodes.orientation.data()->data(),  // input: initial node orientation
                td.blade_nodes_to_blade_num.data(),  // input: blade node to blade number mapping
                &error_handling.error_status,        // output: Error status
                error_handling.error_message.data()  // output: Error message buffer
            );

            error_handling.CheckError();
        }

        //----------------------------------------------------------------------
        // ADI_C_Init
        //----------------------------------------------------------------------

        auto ADI_C_Init = this->lib.get_function<
            void(int*, const char*, int*, int*, const char*, int*, char*, float*, float*, float*, float*, float*, float*, float*, float*, int*, double*, double*, int*, int*, int*, float*, float*, int*, double*, int*, char*, char*, int*, char*)>(
            "ADI_C_Init"
        );

        // Flatten arrays to pass
        auto vtk_nacelle_dim_flat = std::array<float, 6>{};
        std::copy(
            vtk_settings.vtk_nacelle_dimensions.begin(), vtk_settings.vtk_nacelle_dimensions.end(),
            vtk_nacelle_dim_flat.begin()
        );

        // Primary input files will be passed as a single string joined by C_NULL_CHAR i.e. '\0'

        int32_t aerodyn_input_is_passed = this->sim_controls.aerodyn_input_is_path ? 0 : 1;
        std::string aerodyn_input_string = this->sim_controls.aerodyn_input + '\0';
        int32_t aerodyn_input_string_length = static_cast<int>(aerodyn_input_string.size());

        int32_t inflowwind_input_is_passed = this->sim_controls.inflowwind_input_is_path ? 0 : 1;
        std::string inflowwind_input_string = this->sim_controls.inflowwind_input + '\0';
        int32_t inflowwind_input_string_length = static_cast<int>(inflowwind_input_string.size());

        ADI_C_Init(
            &aerodyn_input_is_passed,              // input: AD input is passed
            aerodyn_input_string.data(),           // input: AD input file as string
            &aerodyn_input_string_length,          // input: AD input file string length
            &inflowwind_input_is_passed,           // input: IfW input is passed
            inflowwind_input_string.data(),        // input: IfW input file as string
            &inflowwind_input_string_length,       // input: IfW input file string length
            sim_controls.output_root_name.data(),  // input: rootname for ADI file writing
            &env_conditions.gravity,               // input: gravity
            &air.density,                          // input: air density
            &air.kinematic_viscosity,              // input: kinematic viscosity
            &air.sound_speed,                      // input: speed of sound
            &env_conditions.atm_pressure,          // input: atmospheric pressure
            &air.vapor_pressure,                   // input: vapor pressure
            &env_conditions.water_depth,           // input: water depth
            &env_conditions.msl_offset,            // input: MSL to SWL offset
            &sim_controls.interpolation_order,     // input: interpolation order
            &sim_controls.time_step,               // input: time step
            &sim_controls.max_time,                // input: maximum simulation time
            &sim_controls.store_HH_wind_speed,     // input: store HH wind speed
            &vtk_settings.write_vtk,               // input: write VTK output
            &vtk_settings.vtk_type,                // input: VTK output type
            vtk_nacelle_dim_flat.data(),           // input: VTK nacelle dimensions
            &vtk_settings.vtk_hub_radius,          // input: VTK hub radius
            &sim_controls.output_format,           // input: output format
            &sim_controls.output_time_step,        // input: output time step
            &sim_controls.n_channels,              // output: number of channels
            sim_controls.channel_names_c.data(),   // output: output channel names
            sim_controls.channel_units_c.data(),   // output: output channel units
            &error_handling.error_status,          // output: error status
            error_handling.error_message.data()    // output: error message buffer
        );

        error_handling.CheckError();
    }

    // Wrapper for ADI_C_SetRotorMotion routine to set rotor motion i.e. motion of the hub,
    // nacelle, root, and mesh points from the structural mesh
    void SetRotorMotion() {
        auto ADI_C_SetRotorMotion = this->lib.get_function<void(int*, const float*, const double*, const float*, const float*, const float*, const double*, const float*, const float*, const float*, const double*, const float*, const float*, const int*, const float*, const double*, const float*, const float*, int*, char*)>(
            "ADI_C_SetRotorMotion"
        );

        int32_t turbine_number{0};
        for (const auto& td : this->turbines) {
            ++turbine_number;
            ADI_C_SetRotorMotion(
                &turbine_number,                             // input: current turbine number
                td.hub.position.data()->data(),              // input: hub positions
                td.hub.orientation.data()->data(),           // input: hub orientations
                td.hub.velocity.data()->data(),              // input: hub velocities
                td.hub.acceleration.data()->data(),          // input: hub accelerations
                td.nacelle.position.data()->data(),          // input: nacelle positions
                td.nacelle.orientation.data()->data(),       // input: nacelle orientations
                td.nacelle.velocity.data()->data(),          // input: nacelle velocities
                td.nacelle.acceleration.data()->data(),      // input: nacelle accelerations
                td.blade_roots.position.data()->data(),      // input: root positions
                td.blade_roots.orientation.data()->data(),   // input: root orientations
                td.blade_roots.velocity.data()->data(),      // input: root velocities
                td.blade_roots.acceleration.data()->data(),  // input: root accelerations
                &td.blade_nodes.n_mesh_points,               // input: number of mesh points
                td.blade_nodes.position.data()->data(),      // input: mesh positions
                td.blade_nodes.orientation.data()->data(),   // input: mesh orientations
                td.blade_nodes.velocity.data()->data(),      // input: mesh velocities
                td.blade_nodes.acceleration.data()->data(),  // input: mesh accelerations
                &error_handling.error_status,                // output: error status
                error_handling.error_message.data()          // output: error message buffer
            );

            error_handling.CheckError();
        }
    }

    // Wrapper for ADI_C_GetRotorLoads routine to get aerodynamic loads on the rotor
    void GetRotorAerodynamicLoads() {
        auto ADI_C_GetRotorLoads =
            this->lib.get_function<void(int*, int*, float*, int*, char*)>("ADI_C_GetRotorLoads");

        int32_t turbine_number{0};
        for (auto& td : this->turbines) {
            ++turbine_number;

            ADI_C_GetRotorLoads(
                &turbine_number,                      // input: current turbine number
                &td.blade_nodes.n_mesh_points,        // input: number of mesh points
                td.blade_nodes.loads.data()->data(),  // output: mesh force/moment array
                &error_handling.error_status,         // output: error status
                error_handling.error_message.data()   // output: error message buffer
            );

            error_handling.CheckError();
        }
    }

    // Wrapper for ADI_C_CalcOutput routine to calculate output channels at a given time
    void CalculateOutputChannels(double time, std::vector<float>& output_channel_values) {
        auto ADI_C_CalcOutput =
            this->lib.get_function<void(double*, float*, int*, char*)>("ADI_C_CalcOutput");

        // Set up output channel values
        auto output_channel_values_c =
            std::vector<float>(static_cast<size_t>(sim_controls.n_channels));

        // Run ADI_C_CalcOutput
        ADI_C_CalcOutput(
            &time,                               // input: time at which to calculate output forces
            output_channel_values_c.data(),      // output: output channel values
            &error_handling.error_status,        // output: error status
            error_handling.error_message.data()  // output: error message buffer
        );

        error_handling.CheckError();

        // Copy the output channel values back to the original array
        output_channel_values = output_channel_values_c;
    }

    // Wrapper for ADI_C_UpdateStates routine to calculate output forces at a given time
    void UpdateStates(double time, double time_next) {
        auto ADI_C_UpdateStates =
            this->lib.get_function<void(double*, double*, int*, char*)>("ADI_C_UpdateStates");

        // Run ADI_C_UpdateStates
        ADI_C_UpdateStates(
            &time,                               // input: time at which to calculate output forces
            &time_next,                          // input: time T+dt we are stepping to
            &error_handling.error_status,        // output: error status
            error_handling.error_message.data()  // output: error message buffer
        );

        error_handling.CheckError();
    }

    // Wrapper for ADI_C_End routine to end the AeroDyn Inflow library
    void Finalize() {
        auto ADI_C_End = this->lib.get_function<void(int*, char*)>("ADI_C_End");

        // Run ADI_C_End
        ADI_C_End(
            &error_handling.error_status,        // output: error status
            error_handling.error_message.data()  // output: error message buffer
        );

        error_handling.CheckError();
    }
};

}  // namespace openturbine::util
