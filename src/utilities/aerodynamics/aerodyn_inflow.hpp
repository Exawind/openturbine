#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/math/quaternion_operations.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::util {

/**
 * Following contains a C++ wrapper to interact with the AeroDyn/InflowWind (ADI) shared library,
 * originally written in Fortran, that exposes C-bindings for the AeroDyn and InflowWind modules of
 * OpenFAST. This wrapper simplifies interaction with the ADI library (particularly the C-based
 * interface, with inspiration from the python interface), providing a modern interface for
 * OpenTurbine devs to run AeroDyn x InflowWind.
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
 * Canonical workflow for using the AeroDyn x InflowWind C++ wrapper (see unit test directory for
 * example):
 *   1.  Instantiate the AeroDynInflowLibrary class
 *           - Modify the settings from provided defaults as needed
 *           - Set input files for AeroDyn and InflowWind (from file or as strings)
 *   2.  Initialize the AeroDyn Fortran library via the following steps:
 *           - PreInitialize()  -- pre-initialize the AeroDyn library i.e. set number of turbines
 *           - SetupRotor()     -- set up rotor-specific parameters i.e. initialize one rotor
 *                                  (iterate over turbines)
 *           - Initialize()     -- initialize the AeroDyn library with input files i.e. actually
 *                                  call ADI to initialize the simulation
 *   3.  Perform the simulation by iterating over timesteps:
 *           - SetupRotorMotion()         -- set motions of single turbine (iterate over turbines)
 *           - UpdateStates()             -- update to next time step
 *           - CalculateOutputChannels()  -- get outputs
 *           - GetRotorAerodynamicLoads() -- get loads per rotor (iterate over turbines)
 *   4. Complete the simulation
 *         - Call Finalize() to close the AeroDyn library and free memory
 *         - Handle any resulting errors
 *
 * References:
 * - OpenFAST/AeroDyn:
 *   https://github.com/OpenFAST/openfast/tree/main/modules/aerodyn
 * - AeroDyn InflowWind C bindings:
 *   https://github.com/OpenFAST/openfast/blob/dev/modules/aerodyn/src/AeroDyn_Inflow_C_Binding.f90
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

    static constexpr size_t kErrorMessagesLength{1025U};  //< Max error message length
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

/// Function to break apart a 7x1 generalized coords vector into position (3x1 vector) and
/// orientation (9x1 vector) components
static void SetPositionAndOrientation(
    const std::array<double, 7>& data, std::array<float, 3>& position,
    std::array<double, 9>& orientation
) {
    // Set position (first 3 elements)
    for (size_t i = 0; i < 3; ++i) {
        position[i] = static_cast<float>(data[i]);
    }

    // Set orientation (convert last 4 elements to 3x3 matrix)
    auto orientation_2D =
        QuaternionToRotationMatrix(std::array<double, 4>{data[3], data[4], data[5], data[6]});

    // Flatten the 3x3 matrix to a 1D array
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            orientation[i * 3 + j] = orientation_2D[i][j];
        }
    }
}

/// Struct to hold the initial settings for the turbine
struct TurbineSettings {
    int n_turbines{1};                                      //< Number of turbines - 1 by default
    int n_blades{3};                                        //< Number of blades - 3 by default
    std::array<float, 3> initial_hub_position{0.};          //< Initial hub position
    std::array<double, 9> initial_hub_orientation{0.};      //< Initial hub orientation
    std::array<float, 3> initial_nacelle_position{0.};      //< Initial nacelle position
    std::array<double, 9> initial_nacelle_orientation{0.};  //< Initial nacelle orientation
    std::vector<std::array<float, 3>> initial_root_position{//< Initial root positions of blades
                                                            static_cast<size_t>(n_blades)};
    std::vector<std::array<double, 9>> initial_root_orientation{//< Initial root orientations
                                                                static_cast<size_t>(n_blades)};

    /// Default constructor
    TurbineSettings();

    /// Constructor to initialize all data based on provided 7x1 inputs
    TurbineSettings(
        const std::array<double, 7>& hub_data, const std::array<double, 7>& nacelle_data,
        const std::vector<std::array<double, 7>>& root_data, int n_turbines = 1, int n_blades = 3
    )
        : n_turbines(n_turbines),
          n_blades(n_blades),
          initial_root_position(static_cast<size_t>(n_blades)),
          initial_root_orientation(static_cast<size_t>(n_blades)) {
        // Set hub position and orientation
        SetPositionAndOrientation(hub_data, initial_hub_position, initial_hub_orientation);

        // Set nacelle position and orientation
        SetPositionAndOrientation(
            nacelle_data, initial_nacelle_position, initial_nacelle_orientation
        );

        // Set root positions and orientations
        for (size_t i = 0; i < static_cast<size_t>(n_blades); ++i) {
            SetPositionAndOrientation(
                root_data[i], initial_root_position[i], initial_root_orientation[i]
            );
        }
    };
};

/// Struct to hold the settings for simulation controls
struct SimulationControls {
    static constexpr size_t kDefaultStringLength{1025};  //< Max length for output filenames

    // Input file handling
    int aerodyn_input_passed{1};     //< Input file passed for AeroDyn module? (1: passed)
    int inflowwind_input_passed{1};  //< Input file passed for InflowWind module (1: passed)

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

/// Struct to hold the initial motion of the structural mesh
struct StructuralMesh {
    int n_mesh_points{1};                                       //< Number of mesh points
    std::vector<std::array<float, 3>> initial_mesh_position{};  //< N x 3 array [x, y, z]
    std::vector<std::array<double, 9>>
        initial_mesh_orientation{};  //< N x 9 array [r11, r12, ..., r33]
    std::vector<int>
        mesh_point_to_blade_num{};  //< N x 1 array for mapping a mesh point to blade number

    /// Default constructor
    StructuralMesh() = default;

    /// Constructor to initialize all data based on provided 7x1 inputs
    StructuralMesh(const std::vector<std::array<double, 7>>& mesh_data, int n_mesh_points = 1)
        : n_mesh_points(n_mesh_points),
          initial_mesh_position(static_cast<size_t>(n_mesh_points)),
          initial_mesh_orientation(static_cast<size_t>(n_mesh_points)),
          mesh_point_to_blade_num(static_cast<size_t>(n_mesh_points)) {
        // Set mesh position and orientation
        for (size_t i = 0; i < static_cast<size_t>(n_mesh_points); ++i) {
            SetPositionAndOrientation(
                mesh_data[i], initial_mesh_position[i], initial_mesh_orientation[i]
            );
        }
    }
};

struct MeshMotionData {
    std::vector<std::array<float, 3>> position;      //< N x 3 array [x, y, z]
    std::vector<std::array<double, 9>> orientation;  //< N x 9 array [r11, r12, ..., r33]
    std::vector<std::array<float, 6>> velocity;      //< N x 6 array [u, v, w, p, q, r]
    std::vector<std::array<float, 6>>
        acceleration;  //< N x 6 array [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]

    /// Default constructor
    MeshMotionData() = default;

    /// Constructor to initialize all data based on provided 7x1 inputs
    MeshMotionData(const std::vector<std::array<double, 7>>& mesh_data, int n_mesh_points = 1)
        : position(static_cast<size_t>(n_mesh_points)),
          orientation(static_cast<size_t>(n_mesh_points)),
          velocity(static_cast<size_t>(n_mesh_points)),
          acceleration(static_cast<size_t>(n_mesh_points)) {
        // Set mesh position and orientation
        for (size_t i = 0; i < static_cast<size_t>(n_mesh_points); ++i) {
            SetPositionAndOrientation(mesh_data[i], position[i], orientation[i]);
        }
    }
    /// Method to check the sizes of the input arrays
    template <typename T, size_t N>
    void CheckArraySize(
        const std::vector<std::array<T, N>>& array, size_t expected_rows, size_t expected_cols,
        const std::string& array_name, const std::string& node_type
    ) const {
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

    /// Method to check the input motions i.e. position, orientation, velocity, and acceleration
    /// arrays
    void CheckInputMotions(
        const std::string& node_type, size_t expected_position_dim, size_t expected_orientation_dim,
        size_t expected_vel_acc_dim, size_t expected_number_of_nodes
    ) const {
        CheckArraySize(
            position, expected_number_of_nodes, expected_position_dim, "positions", node_type
        );
        CheckArraySize(
            orientation, expected_number_of_nodes, expected_orientation_dim, "orientations",
            node_type
        );
        CheckArraySize(
            velocity, expected_number_of_nodes, expected_vel_acc_dim, "velocities", node_type
        );
        CheckArraySize(
            acceleration, expected_number_of_nodes, expected_vel_acc_dim, "accelerations", node_type
        );
    }

    /// Method to check the hub/nacelle input motions
    void CheckHubNacelleInputMotions(const std::string& node_name) const {
        const size_t expected_position_dim{3};
        const size_t expected_orientation_dim{9};
        const size_t expected_vel_acc_dim{6};
        const size_t expected_number_of_nodes{1};  // Since there is only 1 hub/nacelle node

        CheckInputMotions(
            node_name, expected_position_dim, expected_orientation_dim, expected_vel_acc_dim,
            expected_number_of_nodes
        );
    }

    /// Method to check the root input motions
    void CheckRootInputMotions(size_t num_blades, size_t init_num_blades) const {
        if (num_blades != init_num_blades) {
            std::cerr << "The number of root points changed from the initial value of "
                      << init_num_blades << ". This is not permitted during the simulation."
                      << std::endl;
            // call ADI_End();
        }

        const size_t expected_position_dim{3};
        const size_t expected_orientation_dim{9};
        const size_t expected_vel_acc_dim{6};

        CheckInputMotions(
            "root", expected_position_dim, expected_orientation_dim, expected_vel_acc_dim, num_blades
        );
    }

    /// Method to check the mesh input motions
    void CheckMeshInputMotions(size_t num_mesh_pts, size_t init_num_mesh_pts) const {
        if (num_mesh_pts != init_num_mesh_pts) {
            std::cerr << "The number of mesh points changed from the initial value of "
                      << init_num_mesh_pts << ". This is not permitted during the simulation."
                      << std::endl;
            // call ADI_End();
        }

        const size_t expected_position_dim{3};
        const size_t expected_orientation_dim{9};
        const size_t expected_vel_acc_dim{6};

        CheckInputMotions(
            "mesh", expected_position_dim, expected_orientation_dim, expected_vel_acc_dim,
            num_mesh_pts
        );
    }
};

/**
 * @brief Wrapper class for the AeroDynInflow (ADI) shared library
 *
 * @details This class provides an interface for interacting with the AeroDynInflow (ADI) shared
 * library, which is a Fortran library offering C bindings for the AeroDyn x InflowWind modules
 * of OpenFAST.
 *
 * Following functions are wrapped in this class:
 *  - PreInitialize -> ADI_C_PreInit: Handles the pre-initialization of the AeroDynInflow module
 *  - SetupRotor -> ADI_C_SetupRotor: Configures rotor-specific parameters before simulation
 *  - Initialize -> ADI_C_Init: Initializes the AeroDynInflow module for simulation
 *  - SetupRotorMotion -> ADI_C_SetRotorMotion: Sets rotor motion for a given time step
 *  - GetRotorAerodynamicLoads -> ADI_C_GetRotorLoads: Gets aerodynamic loads on the rotor
 *  - CalculateOutputChannels -> ADI_C_CalcOutput: Calculates output channels at a given time
 *  - UpdateStates -> ADI_C_UpdateStates: Updates states for the next time step
 *  - Finalize -> ADI_C_End: Ends the AeroDynInflow module by freeing memory
 */
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
    void PreInitialize() {
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
    void SetupRotor(int turbine_number, int is_horizontal_axis, std::vector<float> turbine_ref_pos) {
        auto ADI_C_SetupRotor = this->lib.get_function<
            void(int*, int*, float*, float*, double*, float*, double*, int*, float*, double*, int*, float*, double*, int*, int*, char*)>(
            "ADI_C_SetupRotor"
        );

        // Flatten arrays to pass to the Fortran routine
        auto initial_root_position_flat = FlattenArray(turbine_settings.initial_root_position);
        auto initial_root_orientation_flat = FlattenArray(turbine_settings.initial_root_orientation);
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
            ),                                     // input: initial nacelle orientation
            &turbine_settings.n_blades,            // input: number of blades
            initial_root_position_flat.data(),     // input: initial blade root positions
            initial_root_orientation_flat.data(),  // input: initial blade root orientation
            &structural_mesh.n_mesh_points,        // input: number of mesh points
            init_mesh_pos_flat.data(),             // input: initial node positions
            init_mesh_orient_flat.data(),          // input: initial node orientation
            structural_mesh.mesh_point_to_blade_num.data(
            ),                                   // input: initial mesh point to blade number mapping
            &error_handling.error_status,        // output: Error status
            error_handling.error_message.data()  // output: Error message buffer
        );

        error_handling.CheckError();
    }

    /// Wrapper for ADI_C_Init routine to initialize the AeroDyn Inflow library
    void Initialize(
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

    // Wrapper for ADI_C_SetRotorMotion routine to set rotor motion i.e. motion of the hub,
    // nacelle, root, and mesh points from the structural mesh
    void SetupRotorMotion(
        int turbine_number, MeshMotionData hub_motion, MeshMotionData nacelle_motion,
        MeshMotionData root_motion, MeshMotionData mesh_motion
    ) {
        auto ADI_C_SetRotorMotion = this->lib.get_function<
            void(int*, float*, double*, float*, float*, float*, double*, float*, float*, float*, double*, float*, float*, int*, float*, double*, float*, float*, int*, char*)>(
            "ADI_C_SetRotorMotion"
        );

        // Check the input motions for hub, nacelle, root, and mesh points
        hub_motion.CheckHubNacelleInputMotions("hub");
        nacelle_motion.CheckHubNacelleInputMotions("nacelle");
        root_motion.CheckRootInputMotions(
            static_cast<size_t>(turbine_settings.n_blades),
            static_cast<size_t>(turbine_settings.initial_root_position.size())
        );
        mesh_motion.CheckMeshInputMotions(
            static_cast<size_t>(structural_mesh.n_mesh_points),
            static_cast<size_t>(structural_mesh.initial_mesh_position.size())
        );

        // Flatten the arrays to pass to the Fortran routine
        auto hub_pos_flat = FlattenArray(hub_motion.position);
        auto hub_orient_flat = FlattenArray(hub_motion.orientation);
        auto hub_vel_flat = FlattenArray(hub_motion.velocity);
        auto hub_acc_flat = FlattenArray(hub_motion.acceleration);

        auto nacelle_pos_flat = FlattenArray(nacelle_motion.position);
        auto nacelle_orient_flat = FlattenArray(nacelle_motion.orientation);
        auto nacelle_vel_flat = FlattenArray(nacelle_motion.velocity);
        auto nacelle_acc_flat = FlattenArray(nacelle_motion.acceleration);

        auto root_pos_flat = FlattenArray(root_motion.position);
        auto root_orient_flat = FlattenArray(root_motion.orientation);
        auto root_vel_flat = FlattenArray(root_motion.velocity);
        auto root_acc_flat = FlattenArray(root_motion.acceleration);

        auto mesh_pos_flat = FlattenArray(mesh_motion.position);
        auto mesh_orient_flat = FlattenArray(mesh_motion.orientation);
        auto mesh_vel_flat = FlattenArray(mesh_motion.velocity);
        auto mesh_acc_flat = FlattenArray(mesh_motion.acceleration);

        ADI_C_SetRotorMotion(
            &turbine_number,                     // input: current turbine number
            hub_pos_flat.data(),                 // input: hub positions
            hub_orient_flat.data(),              // input: hub orientations
            hub_vel_flat.data(),                 // input: hub velocities
            hub_acc_flat.data(),                 // input: hub accelerations
            nacelle_pos_flat.data(),             // input: nacelle positions
            nacelle_orient_flat.data(),          // input: nacelle orientations
            nacelle_vel_flat.data(),             // input: nacelle velocities
            nacelle_acc_flat.data(),             // input: nacelle accelerations
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

    // Wrapper for ADI_C_GetRotorLoads routine to get aerodynamic loads on the rotor
    void GetRotorAerodynamicLoads(
        int turbine_number, std::vector<std::array<float, 6>>& mesh_force_moment
    ) {
        auto ADI_C_GetRotorLoads =
            this->lib.get_function<void(int*, int*, float*, int*, char*)>("ADI_C_GetRotorLoads");

        // Flatten the mesh force/moment array
        auto mesh_force_moment_flat = FlattenArray(mesh_force_moment);

        ADI_C_GetRotorLoads(
            &turbine_number,                     // input: current turbine number
            &structural_mesh.n_mesh_points,      // input: number of mesh points
            mesh_force_moment_flat.data(),       // output: mesh force/moment array
            &error_handling.error_status,        // output: error status
            error_handling.error_message.data()  // output: error message buffer
        );

        error_handling.CheckError();

        // Copy the flattened array back to the original array
        for (size_t i = 0; i < mesh_force_moment.size(); ++i) {
            for (size_t j = 0; j < 6; ++j) {
                mesh_force_moment[i][j] = mesh_force_moment_flat[i * 6 + j];
            }
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

    /// Template method to validate array size and flatten it
    template <typename T, size_t N>
    std::vector<T> ValidateAndFlattenArray(
        const std::vector<std::array<T, N>>& array, size_t num_pts, const std::string& array_name
    ) {
        if (array.size() != num_pts) {
            std::cerr << "The number of mesh points in the " << array_name
                      << " array changed from the initial value of " << num_pts
                      << ". This is not permitted during the simulation." << std::endl;
            // call ADI_End();
        }
        return FlattenArray(array);
    }

    /// Flatten and validate position array
    std::vector<float> FlattenPositionArray(
        const std::vector<std::array<float, 3>>& position_array, size_t num_pts
    ) {
        return ValidateAndFlattenArray(position_array, num_pts, "position");
    }

    /// Flatten and validate orientation array
    std::vector<double> FlattenOrientationArray(
        const std::vector<std::array<double, 9>>& orientation_array, size_t num_pts
    ) {
        return ValidateAndFlattenArray(orientation_array, num_pts, "orientation");
    }

    /// Flatten and validate velocity array
    std::vector<float> FlattenVelocityArray(
        const std::vector<std::array<float, 6>>& velocity_array, size_t num_pts
    ) {
        return ValidateAndFlattenArray(velocity_array, num_pts, "velocity");
    }

    /// Method to join a vector of strings into a single string with a delimiter
    std::string JoinStringArray(const std::vector<std::string>& input, char delimiter) {
        std::string output;
        for (const auto& str : input) {
            output += str + delimiter;
        }
        return output;
    }
};

}  // namespace openturbine::util
