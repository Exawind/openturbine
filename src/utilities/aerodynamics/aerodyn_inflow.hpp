#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/math/quaternion_operations.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::util {

/**
 * AeroDynInflowLibrary: C++ wrapper for the AeroDyn/InflowWind (ADI) shared library
 *
 * Provides a modern C++ interface to AeroDyn x InflowWind functionality, encapsulating
 * the C-bindings of the Fortran library.
 *
 * Key Features:
 * - AeroDyn: Blade element momentum (BEM) theory for aerodynamic force calculations
 *   (dynamic stall, unsteady aerodynamics, tower shadow, wind shear, tip/hub losses)
 * - InflowWind: Simulates complex inflow conditions (turbulence, wind shear, gusts)
 *
 * Usage:
 * 1. Instantiate AeroDynInflowLibrary with shared library path
 * 2. Initialize: PreInitialize() -> SetupRotor() -> Initialize()
 * 3. Simulate: SetupRotorMotion() -> UpdateStates() -> CalculateOutputChannels() ->
 *              GetRotorAerodynamicLoads()
 * 4. Finalize() and handle errors
 *
 * See unit tests for detailed usage examples.
 *
 * References:
 * - OpenFAST/AeroDyn: https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html
 * - OpenFAST/InflowWind: https://openfast.readthedocs.io/en/main/source/user/inflowwind/index.html
 * - C bindings:
 * https://github.com/OpenFAST/openfast/blob/dev/modules/aerodyn/src/AeroDyn_Inflow_C_Binding.f90
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

    // Set orientation (convert last 4 elements i.e. quaternion to 3x3 rotation matrix)
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

    /**
     * @brief Validates the turbine settings
     *
     * @details This method validates the turbine settings, including the number of blades, the
     * initial root positions, and the initial root orientations.
     */
    void Validate() const {
        if (n_blades < 1) {
            throw std::runtime_error("No blades. Set n_blades to number of AD blades in the model");
        }

        if (initial_root_position.size() != static_cast<size_t>(n_blades) ||
            initial_root_orientation.size() != static_cast<size_t>(n_blades)) {
            throw std::invalid_argument(
                "Number of blade root positions and orientations must match n_blades"
            );
        }
    }
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

    /**
     * @brief Validates the structural mesh
     *
     * @details This method validates the structural mesh by checking:
     * - The number of mesh points matches the size of initial position and orientation arrays
     * - The size of the mesh point to blade number mapping matches the number of mesh points
     * - All blade numbers in the mapping are valid (between 1 and the number of blades)
     */
    void Validate() const {
        if (initial_mesh_position.size() != static_cast<size_t>(n_mesh_points) ||
            initial_mesh_orientation.size() != static_cast<size_t>(n_mesh_points)) {
            throw std::invalid_argument(
                "Number of mesh positions and orientations must match n_mesh_points"
            );
        }

        if (mesh_point_to_blade_num.size() != static_cast<size_t>(n_mesh_points)) {
            throw std::invalid_argument("Size of mesh_point_to_blade_num must match n_mesh_points");
        }
    }
};

/**
 * @brief Struct to hold the motion + loads data of any structural mesh component
 *
 * @details This struct holds the motion data (i.e. position, orientation,
 * velocity, and acceleration) and loads of the structural mesh, which can be
 * the hub, nacelle, root, or mesh points/nodes
 */
struct MeshData {
    size_t n_mesh_points;                            //< Number of mesh points (nodes)
    std::vector<std::array<float, 3>> position;      //< N x 3 array [x, y, z]
    std::vector<std::array<double, 9>> orientation;  //< N x 9 array [r11, r12, ..., r33]
    std::vector<std::array<float, 6>> velocity;      //< N x 6 array [u, v, w, p, q, r]
    std::vector<std::array<float, 6>>
        acceleration;  //< N x 6 array [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
    std::vector<std::array<float, 6>> loads;  //< N x 6 array [Fx, Fy, Fz, Mx, My, Mz]

    /// Constructor to initialize all mesh data to zero based on provided number of nodes
    MeshData(size_t n_nodes)
        : n_mesh_points(n_nodes),
          position(std::vector<std::array<float, 3>>(n_nodes, {0.f, 0.f, 0.f})),
          orientation(
              std::vector<std::array<double, 9>>(n_nodes, {0., 0., 0., 0., 0., 0., 0., 0., 0.})
          ),
          velocity(std::vector<std::array<float, 6>>(n_nodes, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f})),
          acceleration(std::vector<std::array<float, 6>>(n_nodes, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f})),
          loads(std::vector<std::array<float, 6>>(n_nodes, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f})) {}

    /// Constructor to initialize all mesh data based on provided inputs
    MeshData(
        size_t n_mesh_points, const std::vector<std::array<double, 7>>& mesh_data,
        const std::vector<std::array<float, 6>>& velocities,
        const std::vector<std::array<float, 6>>& accelerations,
        const std::vector<std::array<float, 6>>& loads
    )
        : n_mesh_points(n_mesh_points),
          position(std::vector<std::array<float, 3>>(n_mesh_points, {0.f, 0.f, 0.f})),
          orientation(
              std::vector<std::array<double, 9>>(n_mesh_points, {0., 0., 0., 0., 0., 0., 0., 0., 0.})
          ),
          velocity(std::move(velocities)),
          acceleration(std::move(accelerations)),
          loads(std::move(loads)) {
        if (mesh_data.size() != n_mesh_points || velocities.size() != n_mesh_points ||
            accelerations.size() != n_mesh_points || loads.size() != n_mesh_points) {
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
        const std::string& node_label, size_t expected_number_of_nodes,
        size_t expected_position_dim = 3, size_t expected_orientation_dim = 9,
        size_t expected_vel_acc_dim = 6
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

    void CheckNodeInputMotions(
        const std::string& node_name, size_t expected_number_of_nodes, size_t initial_number_of_nodes
    ) const {
        if (expected_number_of_nodes != initial_number_of_nodes) {
            throw std::invalid_argument(
                "The number of " + node_name + " points changed from the initial value of " +
                std::to_string(initial_number_of_nodes) + " to " +
                std::to_string(expected_number_of_nodes) +
                ". This is not permitted during the simulation."
            );
        }

        CheckInputMotions(node_name, expected_number_of_nodes);
    }

    void CheckHubNacelleInputMotions(const std::string& node_name) const {
        CheckInputMotions(node_name, 1);
    }

    void CheckRootInputMotions(size_t n_blades, size_t init_n_blades) const {
        CheckNodeInputMotions("root", n_blades, init_n_blades);
    }

    void CheckMeshInputMotions(size_t n_mesh_pts, size_t init_n_mesh_pts) const {
        CheckNodeInputMotions("mesh", n_mesh_pts, init_n_mesh_pts);
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
    bool aerodyn_input_passed{true};     //< Input file passed for AeroDyn module? (1: passed)
    bool inflowwind_input_passed{true};  //< Input file passed for InflowWind module? (1: passed)

    // Interpolation order (must be either 1: linear, or 2: quadratic)
    int interpolation_order{1};  //< Interpolation order - linear by default

    // Initial time related variables
    double time_step{0.1};          //< Simulation timestep (s)
    double max_time{600.};          //< Maximum simulation time (s)
    double total_elapsed_time{0.};  //< Total elapsed time (s)
    int n_time_steps{0};            //< Number of time steps

    // Flags
    bool store_HH_wind_speed{true};  //< Flag to store HH wind speed
    bool transpose_DCM{true};        //< Flag to transpose the direction cosine matrix
    int debug_level{0};              //< Debug level (0-4)

    // Outputs
    int output_format{0};         //< File format for writing outputs
    double output_time_step{0.};  //< Timestep for outputs to file
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
    bool write_vtk{false};                      //< Flag to write VTK output
    int vtk_type{1};                            //< Type of VTK output (1: surface meshes)
    std::array<float, 6> vtk_nacelle_dimensions{//< Nacelle dimensions for VTK rendering
                                                -2.5f, -2.5f, 0.f, 10.f, 5.f, 5.f};
    float vtk_hub_radius{1.5f};  //< Hub radius for VTK rendering
};

/**
 * @brief Flattens a 2D array into a 1D array for Fortran compatibility
 *
 * @details This function flattens a 2D array into a 1D array for Fortran compatibility by
 * inserting each element of the 2D array into the 1D array.
 *
 * @tparam T Type of the elements in the array
 * @tparam N Number of elements in each row of the array
 * @param input 2D array to flatten
 * @return Flattened 1D array
 */
template <typename T, size_t N>
std::vector<T> FlattenArray(const std::vector<std::array<T, N>>& input) {
    std::vector<T> output;
    output.reserve(input.size() * N);
    for (const auto& arr : input) {
        output.insert(output.end(), arr.begin(), arr.end());
    }
    return output;
}

/**
 * @brief Validates a 2D array for the correct number of points and then flattens it to 1D
 *
 * @details This function validates a 2D array for the correct number of points and then
 * flattens it to 1D by inserting each element of the 2D array into the 1D array.
 *
 * @tparam T Type of the elements in the array
 * @tparam N Number of elements in each row of the array
 * @param array 2D array to validate and flatten
 * @param expected_size Expected size of the 2D array
 * @return Flattened 1D array
 */
template <typename T, size_t N>
std::vector<T> ValidateAndFlattenArray(
    const std::vector<std::array<T, N>>& array, size_t expected_size
) {
    std::string array_name;
    if constexpr (std::is_same_v<T, float> && N == 3) {
        array_name = "position";
    } else if constexpr (std::is_same_v<T, double> && N == 9) {
        array_name = "orientation";
    } else if constexpr (std::is_same_v<T, float> && N == 6) {
        array_name = "velocity/acceleration";
    } else {
        array_name = "unknown";
    }

    if (array.size() != expected_size) {
        throw std::runtime_error(
            "The number of mesh points in the " + array_name +
            " array changed from the initial value of " + std::to_string(expected_size) + " to " +
            std::to_string(array.size()) + ". This is not permitted during the simulation."
        );
    }
    return FlattenArray(array);
}

/**
 * @brief Unflattens a 1D array into a 2D array
 *
 * @details This function unflattens a 1D array into a 2D array by inserting each element of
 * the 1D array into the 2D array.
 *
 * @tparam T Type of the elements in the array
 * @tparam N Number of elements in each row of the array
 * @param input 1D array to unflatten
 * @return Unflattened 2D array
 */
template <typename T, size_t N>
std::vector<std::array<T, N>> UnflattenArray(const std::vector<T>& input) {
    std::vector<std::array<T, N>> output(input.size() / N);
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < N; ++j) {
            output[i][j] = input[i * N + j];
        }
    }
    return output;
}

/**
 * @brief Joins a vector of strings into a single string with a delimiter
 *
 * @details This function joins a vector of strings into a single string with a delimiter by
 * inserting each element of the vector into the string.
 *
 * @param input Vector of strings to join
 * @param delimiter Delimiter to insert between the strings
 * @return Joined string
 */
std::string JoinStringArray(const std::vector<std::string>& input, char delimiter) {
    if (input.empty()) {
        return "";
    }

    std::ostringstream result;
    std::copy(
        input.begin(), input.end() - 1,
        std::ostream_iterator<std::string>(result, std::string(1, delimiter).c_str())
    );
    result << input.back();

    return result.str();
}

/**
 * @brief Wrapper class for the AeroDynInflow (ADI) shared library
 *
 * @details This class provides an interface for interacting with the AeroDynInflow (ADI) shared
 * library, which is a Fortran library offering C bindings for the AeroDyn x InflowWind modules
 * of OpenFAST.
 *
 * The class encapsulates key functions for AeroDyn/InflowWind simulation:
 *
 * - PreInitialize (ADI_C_PreInit): Set up general parameters
 * - SetupRotor (ADI_C_SetupRotor): Configure rotor-specific settings
 * - Initialize (ADI_C_Init): Complete initialization with input files
 * - SetupRotorMotion (ADI_C_SetRotorMotion): Update rotor motion for each timestep
 * - UpdateStates (ADI_C_UpdateStates): Advance internal states
 * - CalculateOutputChannels (ADI_C_CalcOutput): Compute output values
 * - GetRotorAerodynamicLoads (ADI_C_GetRotorLoads): Retrieve aerodynamic forces and moments
 * - Finalize (ADI_C_End): Clean up and release resources
 *
 * Usage: Instantiate the class, call functions in the order listed above (iterating over
 * turbines/timesteps as needed), and handle any errors using the ErrorHandling struct.
 */
class AeroDynInflowLibrary {
public:
    /// Constructor to initialize AeroDyn Inflow library with default settings and optional path
    AeroDynInflowLibrary(
        std::string shared_lib_path = "aerodyn_inflow_c_binding.dll",
        ErrorHandling eh = ErrorHandling{}, FluidProperties fp = FluidProperties{},
        EnvironmentalConditions ec = EnvironmentalConditions{},
        TurbineSettings ts = TurbineSettings{}, StructuralMesh sm = StructuralMesh{},
        SimulationControls sc = SimulationControls{}, VTKSettings vtk = VTKSettings{}
    )
        : lib_{shared_lib_path, util::dylib::no_filename_decorations},
          error_handling_(std::move(eh)),
          air_(std::move(fp)),
          env_conditions_(std::move(ec)),
          turbine_settings_(std::move(ts)),
          structural_mesh_(std::move(sm)),
          sim_controls_(std::move(sc)),
          vtk_settings_(std::move(vtk)) {}

    /// Destructor to take care of Fortran-side cleanup if the library is initialized
    ~AeroDynInflowLibrary() noexcept {
        try {
            if (is_initialized_) {
                Finalize();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during AeroDynInflowLibrary destruction: " << e.what() << std::endl;
        }
    }

    /// Getter methods for the private member variables
    const ErrorHandling& GetErrorHandling() const { return error_handling_; }
    const EnvironmentalConditions& GetEnvironmentalConditions() const { return env_conditions_; }
    const FluidProperties& GetFluidProperties() const { return air_; }
    const TurbineSettings& GetTurbineSettings() const { return turbine_settings_; }
    const StructuralMesh& GetStructuralMesh() const { return structural_mesh_; }
    const SimulationControls& GetSimulationControls() const { return sim_controls_; }
    const VTKSettings& GetVTKSettings() const { return vtk_settings_; }

    /**
     * @brief Pre-initializes the AeroDyn Inflow library
     *
     * @details This function pre-initializes the AeroDyn Inflow library by setting up the number
     * of turbines, the number of blades, and the transpose DCM flag.
     */
    void PreInitialize() {
        auto ADI_C_PreInit = lib_.get_function<void(int*, int*, int*, int*, char*)>("ADI_C_PreInit");

        // Convert bool -> int to pass to the Fortran routine
        int transpose_DCM_int = sim_controls_.transpose_DCM ? 1 : 0;

        ADI_C_PreInit(
            &turbine_settings_.n_turbines,        // input: Number of turbines
            &transpose_DCM_int,                   // input: Transpose DCM?
            &sim_controls_.debug_level,           // input: Debug level
            &error_handling_.error_status,        // output: Error status
            error_handling_.error_message.data()  // output: Error message
        );

        error_handling_.CheckError();
    }

    /**
     * @brief Sets up the rotor for the AeroDyn Inflow library
     *
     * @details This function sets up the rotor for the AeroDyn Inflow library by initializing the
     * rotor motion data and passing it to the Fortran routine.
     *
     * @param turbine_number Number of the current turbine
     * @param is_horizontal_axis Flag to indicate if the turbine is a horizontal axis turbine
     * @param turbine_ref_pos Reference position of the turbine
     */
    void SetupRotor(
        int turbine_number, bool is_horizontal_axis, std::array<float, 3> turbine_ref_pos
    ) {
        auto ADI_C_SetupRotor = lib_.get_function<
            void(int*, int*, float*, float*, double*, float*, double*, int*, float*, double*, int*, float*, double*, int*, int*, char*)>(
            "ADI_C_SetupRotor"
        );

        // Validate the turbine settings and structural mesh
        turbine_settings_.Validate();
        structural_mesh_.Validate();

        // Flatten arrays to pass to the Fortran routine
        auto initial_root_position_flat = ValidateAndFlattenArray(
            turbine_settings_.initial_root_position, static_cast<size_t>(turbine_settings_.n_blades)
        );
        auto initial_root_orientation_flat = ValidateAndFlattenArray(
            turbine_settings_.initial_root_orientation,
            static_cast<size_t>(turbine_settings_.n_blades)
        );
        auto init_mesh_pos_flat = ValidateAndFlattenArray(
            structural_mesh_.initial_mesh_position,
            static_cast<size_t>(structural_mesh_.n_mesh_points)
        );
        auto init_mesh_orient_flat = ValidateAndFlattenArray(
            structural_mesh_.initial_mesh_orientation,
            static_cast<size_t>(structural_mesh_.n_mesh_points)
        );

        // Convert bool -> int to pass to the Fortran routine
        int is_horizontal_axis_int = is_horizontal_axis ? 1 : 0;

        ADI_C_SetupRotor(
            &turbine_number,                                // input: current turbine number
            &is_horizontal_axis_int,                        // input: 1: HAWT, 0: VAWT or cross-flow
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

    /**
     * @brief Initializes the AeroDyn Inflow library
     *
     * @details This function initializes the AeroDyn Inflow library by passing the input files and
     * other parameters to the Fortran routine.
     *
     * @param aerodyn_input_string_array Input file for the AeroDyn module
     * @param inflowwind_input_string_array Input file for the InflowWind module
     */
    void Initialize(
        std::vector<std::string> aerodyn_input_string_array,
        std::vector<std::string> inflowwind_input_string_array
    ) {
        auto ADI_C_Init = lib_.get_function<
            void(int*, char**, int*, int*, char**, int*, char*, float*, float*, float*, float*, float*, float*, float*, float*, int*, double*, double*, int*, int*, int*, float*, float*, int*, double*, int*, char*, char*, int*, char*)>(
            "ADI_C_Init"
        );

        // Convert bool -> int to pass to the Fortran routine
        int aerodyn_input_passed_int = sim_controls_.aerodyn_input_passed ? 1 : 0;
        int inflowwind_input_passed_int = sim_controls_.inflowwind_input_passed ? 1 : 0;
        int store_HH_wind_speed_int = sim_controls_.store_HH_wind_speed ? 1 : 0;
        int write_vtk_int = vtk_settings_.write_vtk ? 1 : 0;

        // Primary input files will be passed as a single string joined by C_NULL_CHAR i.e. '\0'
        std::string aerodyn_input_string = JoinStringArray(aerodyn_input_string_array, '\0');
        int aerodyn_input_string_length = static_cast<int>(aerodyn_input_string.size());
        char* aerodyn_input_cstring = aerodyn_input_string.data();

        std::string inflowwind_input_string = JoinStringArray(inflowwind_input_string_array, '\0');
        int inflowwind_input_string_length = static_cast<int>(inflowwind_input_string.size());
        char* inflowwind_input_cstring = inflowwind_input_string.data();

        ADI_C_Init(
            &aerodyn_input_passed_int,                    // input: AD input file is passed?
            &aerodyn_input_cstring,                       // input: AD input file as string array
            &aerodyn_input_string_length,                 // input: AD input file string length
            &inflowwind_input_passed_int,                 // input: IfW input file is passed?
            &inflowwind_input_cstring,                    // input: IfW input file as string array
            &inflowwind_input_string_length,              // input: IfW input file string length
            sim_controls_.output_root_name.data(),        // input: rootname for ADI file writing
            &env_conditions_.gravity,                     // input: gravity
            &air_.density,                                // input: air density
            &air_.kinematic_viscosity,                    // input: kinematic viscosity
            &air_.sound_speed,                            // input: speed of sound
            &env_conditions_.atm_pressure,                // input: atmospheric pressure
            &air_.vapor_pressure,                         // input: vapor pressure
            &env_conditions_.water_depth,                 // input: water depth
            &env_conditions_.msl_offset,                  // input: MSL to SWL offset
            &sim_controls_.interpolation_order,           // input: interpolation order
            &sim_controls_.time_step,                     // input: time step
            &sim_controls_.max_time,                      // input: maximum simulation time
            &store_HH_wind_speed_int,                     // input: store HH wind speed?
            &write_vtk_int,                               // input: write VTK output?
            &vtk_settings_.vtk_type,                      // input: VTK output type
            vtk_settings_.vtk_nacelle_dimensions.data(),  // input: VTK nacelle dimensions
            &vtk_settings_.vtk_hub_radius,                // input: VTK hub radius
            &sim_controls_.output_format,                 // input: output format
            &sim_controls_.output_time_step,              // input: output time step
            &sim_controls_.n_channels,                    // output: number of channels
            sim_controls_.channel_names_c.data(),         // output: output channel names
            sim_controls_.channel_units_c.data(),         // output: output channel units
            &error_handling_.error_status,                // output: error status
            error_handling_.error_message.data()          // output: error message buffer
        );

        error_handling_.CheckError();
        is_initialized_ = true;
    }

    /**
     * @brief Sets up the rotor motion for the AeroDyn Inflow library
     *
     * @details This function sets up the rotor motion for the AeroDyn Inflow library by passing
     * the motion data for the hub, nacelle, root, and mesh points to the Fortran routine.
     *
     * @param turbine_number Number of the current turbine
     * @param hub_motion Motion data for the hub
     * @param nacelle_motion Motion data for the nacelle
     * @param root_motion Motion data for the blade roots
     * @param mesh_motion Motion data for the mesh points
     */
    void SetupRotorMotion(
        int turbine_number, MeshData hub_motion, MeshData nacelle_motion, MeshData root_motion,
        MeshData mesh_motion
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

        // Flatten arrays to pass to the Fortran routine
        auto flatten_and_validate = [](const auto& motion, int n_pts) {
            return std::make_tuple(
                ValidateAndFlattenArray(motion.position, static_cast<size_t>(n_pts)),
                ValidateAndFlattenArray(motion.orientation, static_cast<size_t>(n_pts)),
                ValidateAndFlattenArray(motion.velocity, static_cast<size_t>(n_pts)),
                ValidateAndFlattenArray(motion.acceleration, static_cast<size_t>(n_pts))
            );
        };

        // Hub and nacelle (1 point each)
        auto [hub_pos_flat, hub_orient_flat, hub_vel_flat, hub_acc_flat] =
            flatten_and_validate(hub_motion, 1);
        auto [nacelle_pos_flat, nacelle_orient_flat, nacelle_vel_flat, nacelle_acc_flat] =
            flatten_and_validate(nacelle_motion, 1);

        // Root (n_blades points)
        auto [root_pos_flat, root_orient_flat, root_vel_flat, root_acc_flat] =
            flatten_and_validate(root_motion, turbine_settings_.n_blades);

        // Mesh (n_mesh_points)
        auto [mesh_pos_flat, mesh_orient_flat, mesh_vel_flat, mesh_acc_flat] =
            flatten_and_validate(mesh_motion, structural_mesh_.n_mesh_points);

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

    /**
     * @brief Gets the aerodynamic loads on the rotor
     *
     * @details This function gets the aerodynamic loads on the rotor by passing the mesh
     * force/moment array to the Fortran routine.
     *
     * @param turbine_number Number of the current turbine
     * @param mesh_force_moment Mesh force/moment array
     */
    void GetRotorAerodynamicLoads(
        int turbine_number, std::vector<std::array<float, 6>>& mesh_force_moment
    ) {
        auto ADI_C_GetRotorLoads =
            lib_.get_function<void(int*, int*, float*, int*, char*)>("ADI_C_GetRotorLoads");

        // Ensure the input vector has the correct size
        if (mesh_force_moment.size() != static_cast<size_t>(structural_mesh_.n_mesh_points)) {
            throw std::invalid_argument(
                "mesh_force_moment size (" + std::to_string(mesh_force_moment.size()) +
                ") does not match n_mesh_points (" + std::to_string(structural_mesh_.n_mesh_points) +
                ")"
            );
        }

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
        mesh_force_moment = UnflattenArray<float, 6>(mesh_force_moment_flat);
    }

    /**
     * @brief Calculates the output channels at a given time
     *
     * @details This function calculates the output channels at a given time by passing the time
     * and the output channel values to the Fortran routine.
     *
     * @param time Time at which to calculate the output channels
     * @param output_channel_values Output channel values
     */
    void CalculateOutputChannels(double time, std::vector<float>& output_channel_values) {
        auto ADI_C_CalcOutput =
            lib_.get_function<void(double*, float*, int*, char*)>("ADI_C_CalcOutput");

        // Set up output channel values
        auto output_channel_values_c =
            std::vector<float>(static_cast<size_t>(sim_controls_.n_channels), 0.f);

        ADI_C_CalcOutput(
            &time,                                // input: time at which to calculate output forces
            output_channel_values_c.data(),       // output: output channel values
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();

        // Convert output channel values back into the output vector
        output_channel_values = output_channel_values_c;
    }

    /**
     * @brief Updates the states of the AeroDyn Inflow library
     *
     * @details This function updates the states of the AeroDyn Inflow library by passing the current
     * time and the next time to the Fortran routine.
     *
     * @param current_time Current time
     * @param next_time Next time
     */
    void UpdateStates(double current_time, double next_time) {
        auto ADI_C_UpdateStates =
            lib_.get_function<void(double*, double*, int*, char*)>("ADI_C_UpdateStates");

        ADI_C_UpdateStates(
            &current_time,                        // input: current time
            &next_time,                           // input: next time step
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();
    }

    /**
     * @brief Ends the simulation and frees memory
     *
     * @details This function ends the simulation and frees memory by passing the error status
     * and error message buffer to the Fortran routine.
     */
    void Finalize() {
        auto ADI_C_End = lib_.get_function<void(int*, char*)>("ADI_C_End");

        ADI_C_End(
            &error_handling_.error_status,        // output: error status
            error_handling_.error_message.data()  // output: error message buffer
        );

        error_handling_.CheckError();
    }

private:
    bool is_initialized_{false};              //< Flag to check if the library is initialized
    util::dylib lib_;                         //< Dynamic library object for AeroDyn Inflow
    ErrorHandling error_handling_;            //< Error handling settings
    FluidProperties air_;                     //< Properties of the working fluid (air)
    EnvironmentalConditions env_conditions_;  //< Environmental conditions
    TurbineSettings turbine_settings_;        //< Turbine settings
    StructuralMesh structural_mesh_;          //< Structural mesh data
    SimulationControls sim_controls_;         //< Simulation control settings
    VTKSettings vtk_settings_;                //< VTK output settings
};

}  // namespace openturbine::util
