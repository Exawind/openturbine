#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::util {

/**
 * C++ wrapper for interfacing with the AeroDyn Inflow (ADI) shared library, a Fortran-based
 * library that exposes C-bindings for the AeroDyn and InflowWind modules of OpenFAST. This
 * wrapper simplifies interaction with the ADI library (particularly the C-based interface),
 * providing a user-friendly interface for OpenTurbine developers to run AeroDyn with InflowWind.
 *
 * AeroDyn utilizes blade element momentum (BEM) theory to calculate aerodynamic forces acting
 * on each blade section. It accounts for factors such as:
 *  - Dynamic stall
 *  - Unsteady aerodynamics
 *  - Tower shadow
 *  - Wind shear
 *
 * InflowWind simulates the inflow conditions around wind turbines by modeling spatially and
 * temporally varying wind fields. It enables the simulation of complex wind phenomena, such as:
 *  - Turbulence
 *  - Wind shear
 *  - Gusts
 *  - Free vortex wake
 *
 * References:
 * - AeroDyn InflowWind C bindings:
 *   https://github.com/OpenFAST/openfast/blob/dev/modules/aerodyn/src/AeroDyn_Inflow_C_Binding.f90
 * - AeroDyn InflowWind Python interface:
 *   https://github.com/OpenFAST/openfast/blob/dev/modules/aerodyn/python-lib/aerodyn_inflow_library.py
 */

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

    static constexpr size_t kErrorMessagesLength = 1025U;  //< Max error message length
    int abort_error_level{
        static_cast<int>(ErrorLevel::kFatalError)  //< Error level at which to abort
    };
    int error_status{0};                                     //< Error status
    std::array<char, kErrorMessagesLength> error_message{};  //< Error message buffer

    /// Checks for errors and throws an exception if found - otherwise returns true
    bool CheckError() const {
        return error_status == 0 ? true : throw std::runtime_error(error_message.data());
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

/// Struct to hold the initial settings for the turbine (assuming a single turbine and three blades)
struct TurbineSettings {
    int n_turbines{1};                                      //< Number of turbines - 1 by default
    int n_blades{3};                                        //< Number of blades - 3 by default
    std::array<float, 3> initial_hub_position{0.};          //< Initial hub position
    std::array<double, 9> initial_hub_orientation{0.};      //< Initial hub orientation
    std::array<float, 3> initial_nacelle_position{0.};      //< Initial nacelle position
    std::array<double, 9> initial_nacelle_orientation{0.};  //< Initial nacelle orientation
    std::array<float, 3> initial_root_position{0.};         //< Initial root position
    std::array<double, 9> initial_root_orientation{0.};     //< Initial root orientation
};

/// Struct to hold the settings for the simulation controls
struct SimulationControls {
    static constexpr size_t kDefaultStringLength{1025};  //< Max length for output filenames

    // Input file handling
    int aerodyn_input_passed{true};     //< Flag to check if input file passed for AeroDyn module
    int inflowwind_input_passed{true};  //< Flag to check if input file passed for InflowWind module

    // Interpolation order (must be either 1: linear, or 2: quadratic)
    int interpolation_order{1};  //< Interpolation order - linear by default

    // Initial time related variables
    double time_step{0.1};          //< Simulation timestep (s)
    double max_time{600.};          //< Maximum simulation time (s)
    double total_elapsed_time{0.};  //< Total elapsed time (s)
    int n_time_steps{0};            //< Number of time steps

    // Flags
    int store_HH_wind_speed{1};  //< Flag to store HH wind speed
    int transpose_DCM{1};        //< Flag to transpose the direction cosine matrix
    int debug_level{0};          //< Debug level (0-4)

    // Outputs
    int output_format{0};        //< File format for writing outputs
    float output_time_step{0.};  //< Timestep for outputs to file
    std::array<char, kDefaultStringLength> output_root_name{
        "Output_ADIlib_default"  //< Root name for output files
    };
    int n_channels{0};                              //< Number of channels returned
    std::array<char, 20 * 8000> channel_names_c{};  //< Output channel names
    std::array<char, 20 * 8000> channel_units_c{};  //< Output channel units
};

/// Struct to hold the settings for VTK output
struct VTKSettings {
    int write_vtk{false};                       //< Flag to write VTK output
    int vtk_type{1};                            //< Type of VTK output (1: surface meshes)
    std::array<float, 6> vtk_nacelle_dimensions{//< Nacelle dimensions for VTK rendering
                                                -2.5f, -2.5f, 0.f, 10.f, 5.f, 5.f};
    float vtk_hub_radius{1.5f};  //< Hub radius for VTK rendering
};

/// Struct to hold the structural mesh data
struct StructuralMesh {
    int n_mesh_points{1};                                       //< Number of mesh points
    std::vector<std::array<float, 3>> initial_mesh_position{};  //< N x 3 array [x, y, z]
    std::vector<std::array<double, 9>>
        initial_mesh_orientation{};  //< N x 9 array [r11, r12, ..., r33]
    std::vector<int>
        mesh_point_to_blade_num{};  //< N x 1 array for mapping a mesh point to blade number
};

// Define structures for hub, nacelle, root, and mesh motions
struct MotionData {
    std::vector<float> position;
    std::vector<double> orientation;
    std::vector<float> velocity;
    std::vector<float> acceleration;
};

struct MeshMotionData {
    std::vector<std::array<float, 3>> position;
    std::vector<std::array<double, 9>> orientation;
    std::vector<std::array<float, 3>> velocity;
    std::vector<std::array<float, 3>> acceleration;
};

/// Wrapper class for the AeroDynInflow (ADI) shared library
/// @details The AeroDynInflow (ADI) shared library is a Fortran library that provides C bindings
/// for interfacing with the AeroDyn+InflowWind modules of OpenFAST
struct AeroDynInflowLibrary {
    util::dylib lib{
        "libaerodyn_inflow_c_binding.dylib",
        util::dylib::no_filename_decorations  //< Dynamic library object for AeroDyn Inflow
    };
    ErrorHandling error_handling;            //< Error handling settings
    FluidProperties air;                     //< Properties of the working fluid (air)
    EnvironmentalConditions env_conditions;  //< Environmental conditions
    TurbineSettings turbine_settings;        //< Turbine settings
    StructuralMesh structural_mesh;          //< Structural mesh data
    SimulationControls sim_controls;         //< Simulation control settings
    VTKSettings vtk_settings;                //< VTK output settings

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

    /// Wrapper for ADI_Init routine to initialize the AeroDyn Inflow library
    void ADI_Init(
        std::vector<std::string> aerodyn_input_string_array,
        std::vector<std::string> inflowwind_input_string_array
    ) {
        auto ADI_C_Init =
            this->lib
                .get_function<
                    void(int*, const char*, int*, int*, const char*, int*, char*, float*, float*, float*, float*, float*, float*, float*, float*, int*, double*, double*, int*, int*, int*, float*, float*, int*, float*, int*, char*, char*, int*, char*)>(
                    "ADI_C_Init"
                );

        // Flatten arrays to pass
        auto vtk_nacelle_dim_flat = std::array<float, 6>{};
        std::copy(
            vtk_settings.vtk_nacelle_dimensions.begin(), vtk_settings.vtk_nacelle_dimensions.end(),
            vtk_nacelle_dim_flat.begin()
        );

        // Primary input files will be passed as a single string joined by C_NULL_CHAR i.e. '\0'
        std::string aerodyn_input_string = this->JoinStringArray(aerodyn_input_string_array, '\0');
        aerodyn_input_string = aerodyn_input_string + '\0';
        int aerodyn_input_string_length = static_cast<int>(aerodyn_input_string.size());

        std::string inflowwind_input_string =
            this->JoinStringArray(inflowwind_input_string_array, '\0');
        inflowwind_input_string = inflowwind_input_string + '\0';
        int inflowwind_input_string_length = static_cast<int>(inflowwind_input_string.size());

        ADI_C_Init(
            &sim_controls.aerodyn_input_passed,     // input: AD input file is passed
            aerodyn_input_string.data(),            // input: AD input file as string
            &aerodyn_input_string_length,           // input: AD input file string length
            &sim_controls.inflowwind_input_passed,  // input: IfW input file is passed
            inflowwind_input_string.data(),         // input: IfW input file as string
            &inflowwind_input_string_length,        // input: IfW input file string length
            sim_controls.output_root_name.data(),   // input: rootname for ADI file writing
            &env_conditions.gravity,                // input: gravity
            &air.density,                           // input: air density
            &air.kinematic_viscosity,               // input: kinematic viscosity
            &air.sound_speed,                       // input: speed of sound
            &env_conditions.atm_pressure,           // input: atmospheric pressure
            &air.vapor_pressure,                    // input: vapor pressure
            &env_conditions.water_depth,            // input: water depth
            &env_conditions.msl_offset,             // input: MSL to SWL offset
            &sim_controls.interpolation_order,      // input: interpolation order
            &sim_controls.time_step,                // input: time step
            &sim_controls.max_time,                 // input: maximum simulation time
            &sim_controls.store_HH_wind_speed,      // input: store HH wind speed
            &vtk_settings.write_vtk,                // input: write VTK output
            &vtk_settings.vtk_type,                 // input: VTK output type
            vtk_nacelle_dim_flat.data(),            // input: VTK nacelle dimensions
            &vtk_settings.vtk_hub_radius,           // input: VTK hub radius
            &sim_controls.output_format,            // input: output format
            &sim_controls.output_time_step,         // input: output time step
            &sim_controls.n_channels,               // output: number of channels
            sim_controls.channel_names_c.data(),    // output: output channel names
            sim_controls.channel_units_c.data(),    // output: output channel units
            &error_handling.error_status,           // output: error status
            error_handling.error_message.data()     // output: error message buffer
        );

        error_handling.CheckError();
    }

    // Wrapper for ADI_SetRotorMotion routine to set rotor motion i.e. motion of the hub, nacelle,
    // root, and mesh points from the structural mesh
    void ADI_C_SetRotorMotion(
        int turbine_number, MotionData hub_motion, MotionData nacelle_motion,
        MeshMotionData root_motion, MeshMotionData mesh_motion
    ) {
        auto ADI_C_SetRotorMotion = this->lib.get_function<
            void(int*, float*, double*, float*, float*, float*, double*, float*, float*, float*, double*, float*, float*, int*, float*, double*, float*, float*, int*, char*)>(
            "ADI_C_SetRotorMotion"
        );

        // Flatten root and mesh motion arrays
        auto root_pos_flat = FlattenArray(root_motion.position);
        auto root_orient_flat = FlattenArray(root_motion.orientation);
        auto root_vel_flat = FlattenArray(root_motion.velocity);
        auto root_acc_flat = FlattenArray(root_motion.acceleration);

        auto mesh_pos_flat = FlattenArray(mesh_motion.position);
        auto mesh_orient_flat = FlattenArray(mesh_motion.orientation);
        auto mesh_vel_flat = FlattenArray(mesh_motion.velocity);
        auto mesh_acc_flat = FlattenArray(mesh_motion.acceleration);

        // Checck the input motions
        // CheckHubNacelleInputMotions(
        //     hub_motion.position, hub_motion.orientation, hub_motion.velocity,
        //     hub_motion.acceleration, "hub"
        // );
        // CheckHubNacelleInputMotions(
        //     nacelle_motion.position, nacelle_motion.orientation, nacelle_motion.velocity,
        //     nacelle_motion.acceleration, "nacelle"
        // );
        // CheckRootInputMotions(
        //     root_motion.position, root_motion.orientation, root_motion.velocity,
        //     root_motion.acceleration, structural_mesh.n_mesh_points,
        //     structural_mesh.initial_mesh_position.size()
        // );
        // CheckMeshInputMotions(
        //     mesh_motion.position, mesh_motion.orientation, mesh_motion.velocity,
        //     mesh_motion.acceleration, structural_mesh.n_mesh_points,
        //     structural_mesh.initial_mesh_position.size()
        // );

        ADI_C_SetRotorMotion(
            &turbine_number,                     // input: current turbine number
            hub_motion.position.data(),          // input: hub position
            hub_motion.orientation.data(),       // input: hub orientation
            hub_motion.velocity.data(),          // input: hub velocity
            hub_motion.acceleration.data(),      // input: hub acceleration
            nacelle_motion.position.data(),      // input: nacelle position
            nacelle_motion.orientation.data(),   // input: nacelle orientation
            nacelle_motion.velocity.data(),      // input: nacelle velocity
            nacelle_motion.acceleration.data(),  // input: nacelle acceleration
            root_pos_flat.data(),                // input: root positions
            root_orient_flat.data(),             // input: root orientations
            root_vel_flat.data(),                // input: root velocities
            root_acc_flat.data(),                // input: root accelerations
            &structural_mesh.n_mesh_points,      // input: number of mesh points
            mesh_pos_flat.data(),                // input: mesh positions
            mesh_orient_flat.data(),             // input: mesh orientations
            mesh_vel_flat.data(),                // input: mesh velocities
            mesh_acc_flat.data(),                // input: mesh accelerations
            &error_handling.error_status,        // output: error status
            error_handling.error_message.data()  // output: error message buffer
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

    std::vector<float> FlattenPositionArray(
        const std::vector<std::vector<float>>& position_array, size_t num_pts
    ) {
        if (position_array.size() != num_pts) {
            std::cerr << "The number of mesh points changed from the initial value of " << num_pts
                      << ". This is not permitted during the simulation." << std::endl;
            // call ADI_End();
        }

        std::vector<float> mesh_pos_flat;
        for (const auto& pos : position_array) {
            mesh_pos_flat.insert(mesh_pos_flat.end(), pos.begin(), pos.end());
        }
        return mesh_pos_flat;
    }

    std::vector<double> FlattenOrientationArray(
        const std::vector<std::vector<double>>& orientation_array, size_t num_pts
    ) {
        if (orientation_array.size() != num_pts) {
            std::cerr << "The number of mesh points changed from the initial value of " << num_pts
                      << ". This is not permitted during the simulation." << std::endl;
            // call ADI_End();
        }

        std::vector<double> mesh_orient_flat;
        for (const auto& orient : orientation_array) {
            mesh_orient_flat.insert(mesh_orient_flat.end(), orient.begin(), orient.end());
        }
        return mesh_orient_flat;
    }

    std::vector<float> FlattenVelocityArray(
        const std::vector<std::vector<float>>& velocity_array, size_t num_pts
    ) {
        if (velocity_array.size() != num_pts) {
            std::cerr << "The number of mesh points changed from the initial value of " << num_pts
                      << ". This is not permitted during the simulation." << std::endl;
            // call ADI_End();
        }

        std::vector<float> mesh_vel_flat;
        for (const auto& vel : velocity_array) {
            mesh_vel_flat.insert(mesh_vel_flat.end(), vel.begin(), vel.end());
        }
        return mesh_vel_flat;
    }

    /// Method to join a vector of strings into a single string with a delimiter
    std::string JoinStringArray(const std::vector<std::string>& input, char delimiter) {
        std::string output;
        for (const auto& str : input) {
            output += str + delimiter;
        }
        return output;
    }

    template <typename T>
    void CheckArraySize(
        const std::vector<std::vector<T>>& array, size_t expected_rows, size_t expected_cols,
        const std::string& array_name, const std::string& node_type
    ) {
        // Check row count first
        if (array.size() != expected_rows) {
            std::cerr << "Expecting a " << expected_rows << "x" << expected_cols << " array of "
                      << node_type << " " << array_name << " with " << expected_rows << " rows."
                      << std::endl;
            // call ADI_End();
        }

        // Check column count only on the first row to avoid redundant checks
        if (!array.empty() && array[0].size() != expected_cols) {
            std::cerr << "Expecting a " << expected_rows << "x" << expected_cols << " array of "
                      << node_type << " " << array_name << " with " << expected_cols << " columns."
                      << std::endl;
            // call ADI_End();
        }
    }

    void CheckInputMotions(
        const std::vector<std::vector<float>>& position_array,
        const std::vector<std::vector<double>>& orientation_array,
        const std::vector<std::vector<float>>& velocity_array,
        const std::vector<std::vector<float>>& accleration_array, const std::string& node_type,
        size_t expected_position_dim, size_t expected_orientation_dim, size_t expected_VelAcceln_dim,
        size_t expected_number_of_nodes
    ) {
        CheckArraySize(
            position_array, expected_number_of_nodes, expected_position_dim, "positions", node_type
        );
        CheckArraySize(
            orientation_array, expected_number_of_nodes, expected_orientation_dim, "orientations",
            node_type
        );
        CheckArraySize(
            velocity_array, expected_number_of_nodes, expected_VelAcceln_dim, "velocities", node_type
        );
        CheckArraySize(
            accleration_array, expected_number_of_nodes, expected_VelAcceln_dim, "accelerations",
            node_type
        );
    }

    void CheckHubNacelleInputMotions(
        const std::vector<std::vector<float>>& hubPos,
        const std::vector<std::vector<double>>& hubOrient,
        const std::vector<std::vector<float>>& hubVel, const std::vector<std::vector<float>>& hubAcc,
        const std::string& nodeName
    ) {
        // Hub/Nacelle specific checks, where dimensions are 3, 9, and 6 for position, orientation,
        // and velocities/accelerations
        const size_t expected_position_dim = 3;
        const size_t expected_orientation_dim = 9;
        const size_t expected_VelAcceln_dim = 6;
        const size_t expected_number_of_nodes = 1;  // Since there is only 1 hub/nacelle node

        CheckInputMotions(
            hubPos, hubOrient, hubVel, hubAcc, nodeName, expected_position_dim,
            expected_orientation_dim, expected_VelAcceln_dim, expected_number_of_nodes
        );
    }

    void CheckRootInputMotions(
        const std::vector<std::vector<float>>& root_pos,
        const std::vector<std::vector<double>>& root_orient,
        const std::vector<std::vector<float>>& root_vel,
        const std::vector<std::vector<float>>& root_acc, size_t num_blades, size_t init_num_blades
    ) {
        if (num_blades != init_num_blades) {
            std::cerr << "The number of root points changed from the initial value of "
                      << init_num_blades << ". This is not permitted during the simulation."
                      << std::endl;
            // call ADI_End();
        }

        const size_t expected_position_dim = 3;
        const size_t expected_orientation_dim = 9;
        const size_t expected_vel_acc_dim = 6;

        CheckInputMotions(
            root_pos, root_orient, root_vel, root_acc, "root", expected_position_dim,
            expected_orientation_dim, expected_vel_acc_dim, num_blades
        );
    }

    void CheckMeshInputMotions(
        const std::vector<std::vector<float>>& mesh_pos,
        const std::vector<std::vector<double>>& mesh_orient,
        const std::vector<std::vector<float>>& mesh_vel,
        const std::vector<std::vector<float>>& mesh_acc, size_t num_mesh_pts,
        size_t init_num_mesh_pts
    ) {
        if (num_mesh_pts != init_num_mesh_pts) {
            std::cerr << "The number of mesh points changed from the initial value of "
                      << init_num_mesh_pts << ". This is not permitted during the simulation."
                      << std::endl;
            // call ADI_End();
        }

        const size_t expected_position_dim = 3;
        const size_t expected_orientation_dim = 9;
        const size_t expected_vel_acc_dim = 6;

        CheckInputMotions(
            mesh_pos, mesh_orient, mesh_vel, mesh_acc, "mesh", expected_position_dim,
            expected_orientation_dim, expected_vel_acc_dim, num_mesh_pts
        );
    }
};

}  // namespace openturbine::util
