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
 * @brief Struct to hold the initial settings/motion for the turbine
 *
 * @details This struct holds the initial settings/motion for the turbine, including the number of
 * turbines, the number of blades, the initial hub position, the initial hub orientation, the initial
 * nacelle position, the initial nacelle orientation, the initial root positions, and the initial
 * root orientations.
 */
struct TurbineSettings {
    int n_turbines{1};                               //< Number of turbines - 1 by default
    int n_blades{3};                                 //< Number of blades - 3 by default
    std::array<float, 3> initial_hub_position{0.f};  //< Initial hub position
    std::array<double, 9> initial_hub_orientation{
        1., 0., 0., 0., 1., 0., 0., 0., 1.  //< Initial hub orientation
    };
    std::array<float, 3> initial_nacelle_position{0.f};  //< Initial nacelle position
    std::array<double, 9> initial_nacelle_orientation{
        1., 0., 0., 0., 1., 0., 0., 0., 1.  //< Initial nacelle orientation
    };
    std::vector<std::array<float, 3>> initial_root_position{
        {0.f, 0.f, 0.f},  // blade 1
        {0.f, 0.f, 0.f},  // blade 2
        {0.f, 0.f, 0.f}   // blade 3
    };                    //< Initial root positions of blades
    std::vector<std::array<double, 9>> initial_root_orientation{
        {1., 0., 0., 0., 1., 0., 0., 0., 1.},  // blade 1
        {1., 0., 0., 0., 1., 0., 0., 0., 1.},  // blade 2
        {1., 0., 0., 0., 1., 0., 0., 0., 1.}   // blade 3
    };                                         //< Initial root orientations of blades

    /// Default constructor
    TurbineSettings() = default;

    /// Constructor to initialize all data based on provided 7x1 inputs
    TurbineSettings(
        const std::array<double, 7>& hub_data, const std::array<double, 7>& nacelle_data,
        const std::vector<std::array<double, 7>>& root_data, int n_turbines = 1, int n_blades = 3
    )
        : n_turbines(n_turbines),
          n_blades(n_blades),
          initial_root_position(static_cast<size_t>(n_blades)),
          initial_root_orientation(static_cast<size_t>(n_blades)) {
        if (root_data.size() != static_cast<size_t>(n_blades)) {
            throw std::invalid_argument("Number of root data entries must match n_blades");
        }

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

/**
 * @brief Struct to hold the initial motion of the structural mesh
 *
 * @details This struct holds the initial motion of the structural mesh, including the number of
 * mesh points i.e. nodes, the initial mesh position, the initial mesh orientation, and the mapping
 * of mesh points to blade numbers.
 */
struct StructuralMesh {
    int n_mesh_points{1};  //< Number of mesh points
    std::vector<std::array<float, 3>> initial_mesh_position{
        {0.f, 0.f, 0.f}  //< N x 3 array [x, y, z]
    };
    std::vector<std::array<double, 9>> initial_mesh_orientation{
        {1., 0., 0., 0., 1., 0., 0., 0., 1.}  //< N x 9 array [r11, r12, ..., r33]
    };
    std::vector<int> mesh_point_to_blade_num{
        1  //< N x 1 array for mapping a mesh point to blade number
    };

    /// Default constructor
    StructuralMesh() = default;

    /// Constructor to initialize all data based on provided 7x1 inputs and mapping of mesh points
    /// to blade numbers
    StructuralMesh(
        const std::vector<std::array<double, 7>>& mesh_data,
        std::vector<int> mesh_point_to_blade_num, int n_mesh_points = 1
    )
        : n_mesh_points(n_mesh_points),
          initial_mesh_position(static_cast<size_t>(n_mesh_points)),
          initial_mesh_orientation(static_cast<size_t>(n_mesh_points)),
          mesh_point_to_blade_num(std::move(mesh_point_to_blade_num)) {
        if (mesh_data.size() != static_cast<size_t>(n_mesh_points) ||
            this->mesh_point_to_blade_num.size() != static_cast<size_t>(n_mesh_points)) {
            throw std::invalid_argument(
                "Number of mesh data entries and mesh point to blade number entries must match "
                "n_mesh_points"
            );
        }

        // Set mesh position and orientation
        for (size_t i = 0; i < static_cast<size_t>(n_mesh_points); ++i) {
            SetPositionAndOrientation(
                mesh_data[i], initial_mesh_position[i], initial_mesh_orientation[i]
            );
        }
    }
};

/**
 * @brief Struct to hold the motion data of any structural mesh component
 *
 * @details This struct holds the motion data (i.e. position, orientation,
 * velocity, and acceleration) of the structural mesh, which can be the hub, nacelle, root, or
 * mesh points/nodes.
 */
struct MeshMotionData {
    std::vector<std::array<float, 3>> position;      //< N x 3 array [x, y, z]
    std::vector<std::array<double, 9>> orientation;  //< N x 9 array [r11, r12, ..., r33]
    std::vector<std::array<float, 6>> velocity;      //< N x 6 array [u, v, w, p, q, r]
    std::vector<std::array<float, 6>>
        acceleration;  //< N x 6 array [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]

    /// Default constructor
    MeshMotionData() = default;

    /// Constructor to initialize all data based on provided inputs
    MeshMotionData(
        const std::vector<std::array<double, 7>>& mesh_data,
        const std::vector<std::array<float, 6>>& mesh_velocities,
        const std::vector<std::array<float, 6>>& mesh_accelerations, size_t n_mesh_points = 1
    )
        : position(n_mesh_points),
          orientation(n_mesh_points),
          velocity(std::move(mesh_velocities)),
          acceleration(std::move(mesh_accelerations)) {
        if (mesh_data.size() != n_mesh_points || mesh_velocities.size() != n_mesh_points ||
            mesh_accelerations.size() != n_mesh_points) {
            throw std::invalid_argument("Input vector sizes must match n_mesh_points");
        }

        // Set mesh position and orientation
        for (size_t i = 0; i < n_mesh_points; ++i) {
            SetPositionAndOrientation(mesh_data[i], position[i], orientation[i]);
        }
    }

    /**
     * @brief Method to check the dimensions of the input arrays
     *
     * @param array The input array to check
     * @param expected_rows The expected number of rows
     * @param expected_cols The expected number of columns
     * @param array_name The name of the array
     * @param node_label The label of the node (e.g., "hub", "nacelle", "root", "mesh")
     */
    template <typename T, size_t N>
    void CheckArraySize(
        const std::vector<std::array<T, N>>& array, size_t expected_rows, size_t expected_cols,
        const std::string& array_name, const std::string& node_label
    ) const {
        if (array.size() != expected_rows) {
            throw std::invalid_argument(
                "Expecting a " + std::to_string(expected_rows) + "x" +
                std::to_string(expected_cols) + " array of " + node_label + " " + array_name +
                " with " + std::to_string(expected_rows) + " rows, but got " +
                std::to_string(array.size()) + " rows."
            );
        }

        if (!array.empty() && array[0].size() != expected_cols) {
            throw std::invalid_argument(
                "Expecting a " + std::to_string(expected_rows) + "x" +
                std::to_string(expected_cols) + " array of " + node_label + " " + array_name +
                " with " + std::to_string(expected_cols) + " columns, but got " +
                std::to_string(array[0].size()) + " columns."
            );
        }
    }

    void CheckInputMotions(
        const std::string& node_label, size_t expected_position_dim, size_t expected_orientation_dim,
        size_t expected_vel_acc_dim, size_t expected_number_of_nodes
    ) const {
        CheckArraySize(
            position, expected_number_of_nodes, expected_position_dim, "positions", node_label
        );
        CheckArraySize(
            orientation, expected_number_of_nodes, expected_orientation_dim, "orientations",
            node_label
        );
        CheckArraySize(
            velocity, expected_number_of_nodes, expected_vel_acc_dim, "velocities", node_label
        );
        CheckArraySize(
            acceleration, expected_number_of_nodes, expected_vel_acc_dim, "accelerations", node_label
        );
    }

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

    void CheckRootInputMotions(size_t num_blades, size_t init_num_blades) const {
        if (num_blades != init_num_blades) {
            throw std::invalid_argument(
                "The number of root points changed from the initial value of " +
                std::to_string(init_num_blades) + " to " + std::to_string(num_blades) +
                ". This is not permitted during the simulation."
            );
        }

        const size_t expected_position_dim{3};
        const size_t expected_orientation_dim{9};
        const size_t expected_vel_acc_dim{6};

        CheckInputMotions(
            "root", expected_position_dim, expected_orientation_dim, expected_vel_acc_dim, num_blades
        );
    }

    void CheckMeshInputMotions(size_t num_mesh_pts, size_t init_num_mesh_pts) const {
        if (num_mesh_pts != init_num_mesh_pts) {
            throw std::invalid_argument(
                "The number of mesh points changed from the initial value of " +
                std::to_string(init_num_mesh_pts) + " to " + std::to_string(num_mesh_pts) +
                ". This is not permitted during the simulation."
            );
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
 * @brief Struct to hold the settings for simulation controls
 *
 * @details This struct holds the settings for simulation controls, including input file
 * handling, interpolation order, time-related variables, and flags.
 */
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
class AeroDynInflowLibrary {
public:
    /// Constructor to initialize AeroDyn Inflow library with default settings and optional path
    AeroDynInflowLibrary(
        std::string shared_lib_path = "", ErrorHandling eh = ErrorHandling{},
        FluidProperties fp = FluidProperties{},
        EnvironmentalConditions ec = EnvironmentalConditions{},
        TurbineSettings ts = TurbineSettings{}, StructuralMesh sm = StructuralMesh{},
        SimulationControls sc = SimulationControls{}, VTKSettings vtk = VTKSettings{}
    )
        : error_handling_(std::move(eh)),
          air_(std::move(fp)),
          env_conditions_(std::move(ec)),
          turbine_settings_(std::move(ts)),
          structural_mesh_(std::move(sm)),
          sim_controls_(std::move(sc)),
          vtk_settings_(std::move(vtk)) {
        if (!shared_lib_path.empty()) {
            lib_ = util::dylib(shared_lib_path, util::dylib::no_filename_decorations);
        }
    }

    /// Getter methods
    const ErrorHandling& GetErrorHandling() const { return error_handling_; }
    const EnvironmentalConditions& GetEnvironmentalConditions() const { return env_conditions_; }
    const FluidProperties& GetFluidProperties() const { return air_; }
    const TurbineSettings& GetTurbineSettings() const { return turbine_settings_; }
    const StructuralMesh& GetStructuralMesh() const { return structural_mesh_; }
    const SimulationControls& GetSimulationControls() const { return sim_controls_; }
    const VTKSettings& GetVTKSettings() const { return vtk_settings_; }

    /// Wrapper for ADI_C_PreInit routine to initialize AeroDyn Inflow library
    void PreInitialize() {
        auto ADI_C_PreInit = lib_.get_function<void(int*, int*, int*, int*, char*)>("ADI_C_PreInit");
        ADI_C_PreInit(
            &turbine_settings_.n_turbines,        // input: Number of turbines
            &sim_controls_.transpose_DCM,         // input: Transpose DCM?
            &sim_controls_.debug_level,           // input: Debug level
            &error_handling_.error_status,        // output: Error status
            error_handling_.error_message.data()  // output: Error message
        );

        error_handling_.CheckError();
    }

    /// Wrapper for ADI_C_SetupRotor routine to set up the rotor
    void SetupRotor(int turbine_number, int is_horizontal_axis, std::vector<float> turbine_ref_pos) {
        auto ADI_C_SetupRotor = lib_.get_function<
            void(int*, int*, float*, float*, double*, float*, double*, int*, float*, double*, int*, float*, double*, int*, int*, char*)>(
            "ADI_C_SetupRotor"
        );

        // Flatten arrays to pass to the Fortran routine
        auto initial_root_position_flat = FlattenArray(turbine_settings_.initial_root_position);
        auto initial_root_orientation_flat =
            FlattenArray(turbine_settings_.initial_root_orientation);
        auto init_mesh_pos_flat = FlattenArray(structural_mesh_.initial_mesh_position);
        auto init_mesh_orient_flat = FlattenArray(structural_mesh_.initial_mesh_orientation);

        ADI_C_SetupRotor(
            &turbine_number,                                // input: current turbine number
            &is_horizontal_axis,                            // input: 1: HAWT, 0: VAWT or cross-flow
            turbine_ref_pos.data(),                         // input: turbine reference position
            turbine_settings_.initial_hub_position.data(),  // input: initial hub position
            turbine_settings_.initial_hub_orientation.data(),   // input: initial hub orientation
            turbine_settings_.initial_nacelle_position.data(),  // input: initial nacelle position
            turbine_settings_.initial_nacelle_orientation.data(
            ),                                     // input: initial nacelle orientation
            &turbine_settings_.n_blades,           // input: number of blades
            initial_root_position_flat.data(),     // input: initial blade root positions
            initial_root_orientation_flat.data(),  // input: initial blade root orientation
            &structural_mesh_.n_mesh_points,       // input: number of mesh points
            init_mesh_pos_flat.data(),             // input: initial node positions
            init_mesh_orient_flat.data(),          // input: initial node orientation
            structural_mesh_.mesh_point_to_blade_num.data(
            ),                              // input: initial mesh point to blade number mapping
            &error_handling_.error_status,  // output: Error status
            error_handling_.error_message.data()  // output: Error message buffer
        );

        error_handling_.CheckError();
    }

    /// Wrapper for ADI_C_Init routine to initialize the AeroDyn Inflow library
    void Initialize(
        std::vector<std::string> aerodyn_input_string_array,
        std::vector<std::string> inflowwind_input_string_array
    ) {
        auto ADI_C_Init =
            lib_
                .get_function<
                    void(int*, const char*, int*, int*, const char*, int*, char*, float*, float*, float*, float*, float*, float*, float*, float*, int*, double*, double*, int*, int*, int*, float*, float*, int*, float*, int*, char*, char*, int*, char*)>(
                    "ADI_C_Init"
                );

        // Flatten arrays to pass
        auto vtk_nacelle_dim_flat = std::array<float, 6>{};
        std::copy(
            vtk_settings_.vtk_nacelle_dimensions.begin(), vtk_settings_.vtk_nacelle_dimensions.end(),
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
            &sim_controls_.aerodyn_input_passed,     // input: AD input file is passed
            aerodyn_input_string.data(),             // input: AD input file as string
            &aerodyn_input_string_length,            // input: AD input file string length
            &sim_controls_.inflowwind_input_passed,  // input: IfW input file is passed
            inflowwind_input_string.data(),          // input: IfW input file as string
            &inflowwind_input_string_length,         // input: IfW input file string length
            sim_controls_.output_root_name.data(),   // input: rootname for ADI file writing
            &env_conditions_.gravity,                // input: gravity
            &air_.density,                           // input: air density
            &air_.kinematic_viscosity,               // input: kinematic viscosity
            &air_.sound_speed,                       // input: speed of sound
            &env_conditions_.atm_pressure,           // input: atmospheric pressure
            &air_.vapor_pressure,                    // input: vapor pressure
            &env_conditions_.water_depth,            // input: water depth
            &env_conditions_.msl_offset,             // input: MSL to SWL offset
            &sim_controls_.interpolation_order,      // input: interpolation order
            &sim_controls_.time_step,                // input: time step
            &sim_controls_.max_time,                 // input: maximum simulation time
            &sim_controls_.store_HH_wind_speed,      // input: store HH wind speed
            &vtk_settings_.write_vtk,                // input: write VTK output
            &vtk_settings_.vtk_type,                 // input: VTK output type
            vtk_nacelle_dim_flat.data(),             // input: VTK nacelle dimensions
            &vtk_settings_.vtk_hub_radius,           // input: VTK hub radius
            &sim_controls_.output_format,            // input: output format
            &sim_controls_.output_time_step,         // input: output time step
            &sim_controls_.n_channels,               // output: number of channels
            sim_controls_.channel_names_c.data(),    // output: output channel names
            sim_controls_.channel_units_c.data(),    // output: output channel units
            &error_handling_.error_status,           // output: error status
            error_handling_.error_message.data()     // output: error message buffer
        );

        error_handling_.CheckError();
    }

    // Wrapper for ADI_C_SetRotorMotion routine to set rotor motion i.e. motion of the hub,
    // nacelle, root, and mesh points from the structural mesh
    void SetupRotorMotion(
        int turbine_number, MeshMotionData hub_motion, MeshMotionData nacelle_motion,
        MeshMotionData root_motion, MeshMotionData mesh_motion
    ) {
        auto ADI_C_SetRotorMotion = lib_.get_function<
            void(int*, float*, double*, float*, float*, float*, double*, float*, float*, float*, double*, float*, float*, int*, float*, double*, float*, float*, int*, char*)>(
            "ADI_C_SetRotorMotion"
        );

        // Check the input motions for hub, nacelle, root, and mesh points
        hub_motion.CheckHubNacelleInputMotions("hub");
        nacelle_motion.CheckHubNacelleInputMotions("nacelle");
        root_motion.CheckRootInputMotions(
            static_cast<size_t>(turbine_settings_.n_blades),
            static_cast<size_t>(turbine_settings_.initial_root_position.size())
        );
        mesh_motion.CheckMeshInputMotions(
            static_cast<size_t>(structural_mesh_.n_mesh_points),
            static_cast<size_t>(structural_mesh_.initial_mesh_position.size())
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
            &turbine_number,                      // input: current turbine number
            hub_pos_flat.data(),                  // input: hub positions
            hub_orient_flat.data(),               // input: hub orientations
            hub_vel_flat.data(),                  // input: hub velocities
            hub_acc_flat.data(),                  // input: hub accelerations
            nacelle_pos_flat.data(),              // input: nacelle positions
            nacelle_orient_flat.data(),           // input: nacelle orientations
            nacelle_vel_flat.data(),              // input: nacelle velocities
            nacelle_acc_flat.data(),              // input: nacelle accelerations
            root_pos_flat.data(),                 // input: root positions
            root_orient_flat.data(),              // input: root orientations
            root_vel_flat.data(),                 // input: root velocities
            root_acc_flat.data(),                 // input: root accelerations
            &structural_mesh_.n_mesh_points,      // input: number of mesh points
            mesh_pos_flat.data(),                 // input: mesh positions
            mesh_orient_flat.data(),              // input: mesh orientations
            mesh_vel_flat.data(),                 // input: mesh velocities
            mesh_acc_flat.data(),                 // input: mesh accelerations
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();
    }

    // Wrapper for ADI_C_GetRotorLoads routine to get aerodynamic loads on the rotor
    void GetRotorAerodynamicLoads(
        int turbine_number, std::vector<std::array<float, 6>>& mesh_force_moment
    ) {
        auto ADI_C_GetRotorLoads =
            lib_.get_function<void(int*, int*, float*, int*, char*)>("ADI_C_GetRotorLoads");

        // Flatten the mesh force/moment array
        auto mesh_force_moment_flat = FlattenArray(mesh_force_moment);

        ADI_C_GetRotorLoads(
            &turbine_number,                      // input: current turbine number
            &structural_mesh_.n_mesh_points,      // input: number of mesh points
            mesh_force_moment_flat.data(),        // output: mesh force/moment array
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();

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
            lib_.get_function<void(double*, float*, int*, char*)>("ADI_C_CalcOutput");

        // Set up output channel values
        auto output_channel_values_c =
            std::vector<float>(static_cast<size_t>(sim_controls_.n_channels));

        // Run ADI_C_CalcOutput
        ADI_C_CalcOutput(
            &time,                                // input: time at which to calculate output forces
            output_channel_values_c.data(),       // output: output channel values
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();

        // Copy the output channel values back to the original array
        output_channel_values = output_channel_values_c;
    }

    // Wrapper for ADI_C_UpdateStates routine to calculate output forces at a given time
    void UpdateStates(double time, double time_next) {
        auto ADI_C_UpdateStates =
            lib_.get_function<void(double*, double*, int*, char*)>("ADI_C_UpdateStates");

        // Run ADI_C_UpdateStates
        ADI_C_UpdateStates(
            &time,                                // input: time at which to calculate output forces
            &time_next,                           // input: time T+dt we are stepping to
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();
    }

    // Wrapper for ADI_C_End routine to end the AeroDyn Inflow library
    void Finalize() {
        auto ADI_C_End = lib_.get_function<void(int*, char*)>("ADI_C_End");

        // Run ADI_C_End
        ADI_C_End(
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();
    }

private:
    util::dylib lib_{
        "libaerodyn_inflow_c_binding.dylib",
        util::dylib::no_filename_decorations  //< Dynamic library object for AeroDyn Inflow
    };
    ErrorHandling error_handling_;            //< Error handling settings
    FluidProperties air_;                     //< Properties of the working fluid (air)
    EnvironmentalConditions env_conditions_;  //< Environmental conditions
    TurbineSettings turbine_settings_;        //< Turbine settings
    StructuralMesh structural_mesh_;          //< Structural mesh data
    SimulationControls sim_controls_;         //< Simulation control settings
    VTKSettings vtk_settings_;                //< VTK output settings

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
