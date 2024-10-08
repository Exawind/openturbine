#pragma once

#include <array>
#include <numeric>
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
 * @brief Configuration for the initial state of a turbine
 *
 * This struct encapsulates the initial configuration data for a wind turbine,
 * including its type, reference position, and initial position of key components
 * in 7x1 arrays [x, y, z, qw, qx, qy, qz]
 */
struct TurbineConfig {
    /**
     * @brief Initial state for a single blade of a turbine
     *
     * Stores the initial position of a blade's root and nodes in 7x1 arrays [x, y, z, qw, qx, qy,
     * qz]
     */
    struct BladeInitialState {
        Array_7 root_initial_position;  //< Initial root position of the blade (1 per blade)
        std::vector<Array_7> node_initial_positions;  //< Initial node positions of the blade

        BladeInitialState(const Array_7& root, const std::vector<Array_7>& nodes)
            : root_initial_position(root), node_initial_positions(nodes) {}
    };

    bool is_horizontal_axis{true};                           //< Is a horizontal axis turbine?
    std::array<float, 3> reference_position{0.f, 0.f, 0.f};  //< Reference position of the turbine
    Array_7 hub_initial_position;                            //< Initial hub position
    Array_7 nacelle_initial_position;                        //< Initial nacelle position
    std::vector<BladeInitialState>
        blade_initial_states;  //< Initial root and node positions of blades (size = n_blades)

    TurbineConfig(
        bool is_hawt, std::array<float, 3> ref_pos, Array_7 hub_pos, Array_7 nacelle_pos,
        std::vector<BladeInitialState> blade_states
    )
        : is_horizontal_axis(is_hawt),
          reference_position(std::move(ref_pos)),
          hub_initial_position(std::move(hub_pos)),
          nacelle_initial_position(std::move(nacelle_pos)),
          blade_initial_states(std::move(blade_states)) {
        // Make sure the initial states are valid
        Validate();
    }

    void Validate() const {
        // Check if there are any blades defined
        if (blade_initial_states.empty()) {
            throw std::runtime_error("No blades defined. At least one blade is required.");
        }

        // Check if there are any nodes defined for each blade
        for (const auto& blade : blade_initial_states) {
            if (blade.node_initial_positions.empty()) {
                throw std::runtime_error(
                    "No nodes defined for a blade. At least one node is required."
                );
            }
        }
    }

    /// Returns the number of blades in the turbine
    size_t NumberOfBlades() const { return blade_initial_states.size(); }
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

    // Set orientation (converts last 4 elements i.e. quaternion -> 3x3 rotation matrix)
    auto orientation_2D = QuaternionToRotationMatrix({data[3], data[4], data[5], data[6]});

    // Flatten the 3x3 matrix to a 1D array
    std::copy(&orientation_2D[0][0], &orientation_2D[0][0] + 9, orientation.begin());
}

/**
 * @brief Struct to hold the motion + loads data of any structural mesh component in
 * AeroDyn/InflowWind compatible format
 *
 * @details This struct holds the motion data (i.e. position 3x1, orientation 9x1,
 * velocity 6x1, and acceleration 6x1) and aerodynamic loads 6x1 of the structural
 *  mesh components-- which can be the hub, nacelle, root, or blade
 */
struct MeshData {
    int32_t n_mesh_points;                           //< Number of mesh points/nodes, N
    std::vector<std::array<float, 3>> position;      //< N x 3 array [x, y, z]
    std::vector<std::array<double, 9>> orientation;  //< N x 9 array [r11, r12, ..., r33]
    std::vector<std::array<float, 6>> velocity;      //< N x 6 array [u, v, w, p, q, r]
    std::vector<std::array<float, 6>>
        acceleration;  //< N x 6 array [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
    std::vector<std::array<float, 6>> loads;  //< N x 6 array [Fx, Fy, Fz, Mx, My, Mz]

    /// Constructor to initialize all mesh data to zero based on provided number of nodes
    MeshData(size_t n_nodes)
        : n_mesh_points(static_cast<int32_t>(n_nodes)),
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
        : n_mesh_points(static_cast<int32_t>(n_mesh_points)),
          position(std::vector<std::array<float, 3>>(n_mesh_points, {0.f, 0.f, 0.f})),
          orientation(
              std::vector<std::array<double, 9>>(n_mesh_points, {0., 0., 0., 0., 0., 0., 0., 0., 0.})
          ),
          velocity(std::move(velocities)),
          acceleration(std::move(accelerations)),
          loads(std::move(loads)) {
        // Set mesh position and orientation from 7x1 array [x, y, z, qw, qx, qy, qz]
        for (size_t i = 0; i < n_mesh_points; ++i) {
            SetPositionAndOrientation(mesh_data[i], position[i], orientation[i]);
        }

        // Make sure the mesh data is valid
        Validate();
    }

    void Validate() const {
        // Check we have at least one node
        if (n_mesh_points <= 0) {
            throw std::invalid_argument("Number of mesh points must be at least 1");
        }

        // Check all vectors are the same size as the number of mesh points
        const size_t expected_size = static_cast<size_t>(n_mesh_points);
        auto check_vector_size = [](const auto& vec, size_t expected_size,
                                    const std::string& vec_name) {
            if (vec.size() != expected_size) {
                throw std::invalid_argument(vec_name + " vector size does not match n_mesh_points");
            }
        };

        check_vector_size(position, expected_size, "Position");
        check_vector_size(orientation, expected_size, "Orientation");
        check_vector_size(velocity, expected_size, "Velocity");
        check_vector_size(acceleration, expected_size, "Acceleration");
        check_vector_size(loads, expected_size, "Loads");
    }

    /// Returns the number of mesh points as size_t
    size_t NumberOfMeshPoints() const { return static_cast<size_t>(n_mesh_points); }
};

/**
 * @brief Struct to hold and manage turbine-specific data
 *
 * @details This struct contains data for a single turbine, including the number of blades,
 * mesh data for various components (hub, nacelle, blade roots, blade nodes), and blade node
 * mappings.
 */
struct TurbineData {
    int32_t n_blades;      //< Number of blades
    MeshData hub;          //< Hub data (1 point)
    MeshData nacelle;      //< Nacelle data (1 point)
    MeshData blade_roots;  //< Blade roots data (n_blades points)
    MeshData blade_nodes;  //< Blade nodes data (sum of nodes per blade)
    /**
     * @brief Mapping of blade nodes to blade numbers (1D array)
     *
     * This vector stores the blade number (1-based index) for each blade node.
     * It allows quick lookup of which blade a particular node belongs to.
     * The size of this vector is equal to the total number of blade nodes across all blades.
     */
    std::vector<int32_t> blade_nodes_to_blade_num_mapping;

    /**
     * @brief Unique indices of nodes for each blade (2D array)
     *
     * This is a vector of vectors where each inner vector contains the unique indices of nodes
     * belonging to a specific blade. The outer vector's size is equal to the number of blades,
     * and each inner vector's size is equal to the number of nodes on that blade.
     * This structure allows quick access to all nodes of a particular blade.
     */
    std::vector<std::vector<size_t>> node_indices_by_blade;

    /**
     * @brief Constructor for TurbineData based on a TurbineConfig object
     *
     * @details This constructor initializes the turbine data in AeroDyn/InflowWind compatible format
     * based on the provided TurbineConfig object in 7x1 arrays i.e. OpenTurbine format.
     *
     * @param tc The TurbineConfig object containing the initial state of the turbine
     */
    TurbineData(const TurbineConfig& tc)
        : n_blades(static_cast<int32_t>(tc.NumberOfBlades())),
          hub(1),
          nacelle(1),
          blade_roots(tc.NumberOfBlades()),
          blade_nodes(CalculateTotalBladeNodes(tc)),
          node_indices_by_blade(tc.NumberOfBlades()) {
        // Initialize hub and nacelle data
        InitializeHubAndNacelle(tc);

        // Initialize blade data
        InitializeBlades(tc);

        // Make sure the turbine data is valid
        Validate();
    }

    void Validate() const {
        // Check if the number of blades is valid
        if (n_blades < 1) {
            throw std::runtime_error("Invalid number of blades. Must be at least 1.");
        }

        // Validate hub and nacelle data - should have exactly one mesh point each
        if (hub.NumberOfMeshPoints() != 1 || nacelle.NumberOfMeshPoints() != 1) {
            throw std::runtime_error("Hub and nacelle should have exactly one mesh point each.");
        }

        // Check if the number of blade roots matches the number of blades
        if (blade_roots.NumberOfMeshPoints() != NumberOfBlades()) {
            throw std::runtime_error("Number of blade roots does not match number of blades.");
        }

        // Check if the blade nodes to blade number mapping is valid - should be the same size
        if (blade_nodes_to_blade_num_mapping.size() !=
            static_cast<size_t>(blade_nodes.NumberOfMeshPoints())) {
            throw std::runtime_error("Blade node to blade number mapping size mismatch.");
        }

        // Check if the node indices by blade are valid - should be same size as number of blades
        if (node_indices_by_blade.size() != NumberOfBlades()) {
            throw std::runtime_error("Node indices by blade size mismatch.");
        }

        // Check if the total number of blade nodes is valid - should be the same as the number of
        // aggregrated blade nodes
        size_t total_nodes = 0;
        for (const auto& bl : node_indices_by_blade) {
            total_nodes += bl.size();
        }
        if (total_nodes != blade_nodes.NumberOfMeshPoints()) {
            throw std::runtime_error("Total number of blade nodes mismatch.");
        }
    }

    /// Returns the number of blades as size_t
    size_t NumberOfBlades() const { return static_cast<size_t>(n_blades); }

    /**
     * @brief Sets the blade node values based on blade number and node number.
     *
     * This method updates the blade node values in the mesh based on the provided blade number and
     * node number. It simplifies the process of updating blade node values by abstracting away the
     * indexing logic.
     *
     * @param blade_number The number of the blade to update.
     * @param node_number The number of the node to update within the specified blade.
     * @param position The new position of the node [x, y, z]
     * @param orientation The new orientation of the node [r11, r12, ..., r33]
     * @param velocity The new velocity of the node [u, v, w, p, q, r]
     * @param acceleration The new acceleration of the node [u_dot, v_dot, w_dot, p_dot, q_dot,
     * r_dot]
     * @param loads The new loads on the node [Fx, Fy, Fz, Mx, My, Mz]
     */
    void SetBladeNodeValues(
        size_t blade_number, size_t node_number, const std::array<float, 3>& position,
        const std::array<double, 9>& orientation, const std::array<float, 6>& velocity,
        const std::array<float, 6>& acceleration, const std::array<float, 6>& loads
    ) {
        // Check if the blade and node numbers are within the valid range
        if (blade_number >= NumberOfBlades() ||
            node_number >= node_indices_by_blade[blade_number].size()) {
            throw std::out_of_range("Blade or node number out of range.");
        }

        // Get the node index and set the values for the node
        size_t node_index = node_indices_by_blade[blade_number][node_number];
        blade_nodes.position[node_index] = position;
        blade_nodes.orientation[node_index] = orientation;
        blade_nodes.velocity[node_index] = velocity;
        blade_nodes.acceleration[node_index] = acceleration;
        blade_nodes.loads[node_index] = loads;
    }

private:
    /// Calculates the total number of blade nodes across all blades
    static size_t CalculateTotalBladeNodes(const TurbineConfig& tc) {
        return std::accumulate(
            tc.blade_initial_states.begin(), tc.blade_initial_states.end(), size_t{0},
            [](size_t sum, const auto& blade) {
                return sum + blade.node_initial_positions.size();
            }
        );
    }

    /// Initializes the hub and nacelle positions and orientations
    void InitializeHubAndNacelle(const TurbineConfig& tc) {
        SetPositionAndOrientation(tc.hub_initial_position, hub.position[0], hub.orientation[0]);
        SetPositionAndOrientation(
            tc.nacelle_initial_position, nacelle.position[0], nacelle.orientation[0]
        );
    }

    /// Initializes blade data including roots, nodes, and mappings
    void InitializeBlades(const TurbineConfig& tc) {
        size_t i_blade{0};
        size_t i_node{0};

        // Initialize blade roots and nodes
        for (const auto& blade : tc.blade_initial_states) {
            // Initialize blade root
            SetPositionAndOrientation(
                blade.root_initial_position, blade_roots.position[i_blade],
                blade_roots.orientation[i_blade]
            );

            // Initialize blade nodes
            for (const auto& node : blade.node_initial_positions) {
                SetPositionAndOrientation(
                    node, blade_nodes.position[i_node], blade_nodes.orientation[i_node]
                );
                node_indices_by_blade[i_blade].emplace_back(i_node);
                blade_nodes_to_blade_num_mapping.emplace_back(static_cast<int32_t>(i_blade + 1));
                ++i_node;
            }
            ++i_blade;
        }
    }
};

/**
 * @brief Struct to hold the settings for simulation controls
 *
 * @details This struct holds the settings for simulation controls, including input file
 * handling, interpolation order, time-related variables, and flags.
 */
struct SimulationControls {
    /// Debug levels used in AeroDyn-Inflow C bindings
    enum class DebugLevel {
        kNone = 0,        //< No debug output
        kSummary = 1,     //< Some summary info
        kDetailed = 2,    //< Above + all position/orientation info
        kInputFiles = 3,  //< Above + input files (if directly passed)
        kAll = 4,         //< Above + meshes
    };

    static constexpr size_t kDefaultStringLength{1025};  //< Max length for output filenames

    // Input file handling
    bool is_aerodyn_input_path{true};     //< Input file passed for AeroDyn module?
    bool is_inflowwind_input_path{true};  //< Input file passed for InflowWind module?
    std::string aerodyn_input;            //< Path to AeroDyn input file
    std::string inflowwind_input;         //< Path to InflowWind input file

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
    int debug_level{static_cast<int>(DebugLevel::kNone)};  //< Debug level (0-4)

    // Outputs
    int output_format{1};                               //< File format for writing outputs
    double output_time_step{0.1};                       //< Timestep for outputs to file
    std::string output_root_name{"ADI_out"};            //< Root name for output files
    int n_channels{0};                                  //< Number of channels returned
    std::array<char, 20 * 8000 + 1> channel_names_c{};  //< Output channel names
    std::array<char, 20 * 8000 + 1> channel_units_c{};  //< Output channel units
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
 * @brief Wrapper class for the AeroDynInflow (ADI) shared library
 *
 * @details This class provides a C++ interface for interacting with the AeroDynInflow (ADI) shared
 * library, which offers C bindings for the AeroDyn x InflowWind modules of OpenFAST.
 * It encapsulates key functions for AeroDyn/InflowWind simulation, including initialization,
 * rotor setup, state updates, and load calculations.
 *
 * Key functions:
 * - Initialize: Set up the simulation environment
 * - SetupRotorMotion: Update rotor motion for each timestep
 * - UpdateStates: Advance internal states
 * - CalculateOutputChannels: Compute output values
 * - GetRotorAerodynamicLoads: Retrieve aerodynamic forces and moments
 * - Finalize: Clean up and release resources
 *
 * @note This class manages the lifecycle of the ADI library, ensuring proper initialization
 * and cleanup.
 */
class AeroDynInflowLibrary {
public:
    /**
     * @brief Construct a new AeroDynInflowLibrary object
     *
     * @param shared_lib_path Path to the ADI shared library
     * @param eh Error handling settings
     * @param fp Fluid properties
     * @param ec Environmental conditions
     * @param sc Simulation control settings
     * @param vtk VTK output settings
     */
    AeroDynInflowLibrary(
        std::string shared_lib_path = "aerodyn_inflow_c_binding.dll",
        ErrorHandling eh = ErrorHandling{}, FluidProperties fp = FluidProperties{},
        EnvironmentalConditions ec = EnvironmentalConditions{},
        SimulationControls sc = SimulationControls{}, VTKSettings vtk = VTKSettings{}
    )
        : lib_{shared_lib_path, util::dylib::no_filename_decorations},
          error_handling_(std::move(eh)),
          air_(std::move(fp)),
          env_conditions_(std::move(ec)),
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
    const SimulationControls& GetSimulationControls() const { return sim_controls_; }
    const VTKSettings& GetVTKSettings() const { return vtk_settings_; }

    /**
     * @brief Initialize the AeroDyn Inflow library
     *
     * @details Performs a complete initialization of the AeroDyn Inflow library through a
     * three-step process:
     * 1. Pre-initialization (ADI_C_PreInit)
     * 2. Rotor setup (ADI_C_SetupRotor)
     * 3. Final initialization (ADI_C_Init)
     *
     * @param turbine_configs Vector of TurbineConfig objects, each representing a single turbine
     */
    void Initialize(std::vector<TurbineConfig> turbine_configs) {
        PreInitialize(turbine_configs.size());
        SetupRotors(turbine_configs);
        FinalizeInitialization();
        is_initialized_ = true;
    }

    /**
     * @brief Performs pre-initialization setup
     *
     * @param n_turbines Number of turbines in the simulation
     */
    void PreInitialize(size_t n_turbines) {
        auto ADI_C_PreInit = lib_.get_function<void(int*, int*, int*, int*, char*)>("ADI_C_PreInit");

        // Convert bool and other types to int32_t for Fortran compatibility
        int32_t debug_level_int = static_cast<int32_t>(sim_controls_.debug_level);
        int32_t transpose_dcm_int = sim_controls_.transpose_DCM ? 1 : 0;
        int32_t n_turbines_int = static_cast<int32_t>(n_turbines);

        ADI_C_PreInit(
            &n_turbines_int,                      // input: Number of turbines
            &transpose_dcm_int,                   // input: Transpose DCM?
            &debug_level_int,                     // input: Debug level
            &error_handling_.error_status,        // output: Error status
            error_handling_.error_message.data()  // output: Error message
        );

        error_handling_.CheckError();
    }

    /**
     * @brief Sets up rotor configurations for all turbines
     *
     * @param turbine_configs Vector of turbine configurations
     */
    void SetupRotors(const std::vector<TurbineConfig>& turbine_configs) {
        auto ADI_C_SetupRotor = lib_.get_function<
            void(int*, int*, const float*, float*, double*, float*, double*, int*, float*, double*, int*, float*, double*, int*, int*, char*)>(
            "ADI_C_SetupRotor"
        );

        // Loop through turbine configurations
        int32_t turbine_number{0};
        for (const auto& tc : turbine_configs) {
            // Turbine number is 1 indexed i.e. 1, 2, 3, ...
            ++turbine_number;

            // Convert bool -> int to pass to the Fortran routine
            int32_t is_horizontal_axis_int = tc.is_horizontal_axis ? 1 : 0;

            // Validate the turbine config
            tc.Validate();

            // Create new turbine data
            // Note: TurbineData and MeshData are validated during construction
            turbines_.emplace_back(TurbineData(tc));
            auto& td = turbines_.back();

            // Call setup rotor for each turbine
            ADI_C_SetupRotor(
                &turbine_number,                            // input: current turbine number
                &is_horizontal_axis_int,                    // input: 1: HAWT, 0: VAWT or cross-flow
                tc.reference_position.data(),               // input: turbine reference position
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
                td.blade_nodes_to_blade_num_mapping.data(
                ),                                    // input: blade node to blade number mapping
                &error_handling_.error_status,        // output: Error status
                error_handling_.error_message.data()  // output: Error message buffer
            );

            error_handling_.CheckError();
        }
    }

    /**
     * @brief Finalizes the initialization process
     */
    void FinalizeInitialization() {
        auto ADI_C_Init =
            lib_.get_function<
                void(int*, char**, int*, int*, char**, int*, const char*, float*, float*, float*, float*, float*, float*, float*, float*, int*, double*, double*, int*, int*, int*, float*, float*, int*, double*, int*, char*, char*, int*, char*)>(
                "ADI_C_Init"
            );

        // Convert bool -> int to pass to the Fortran routine
        int32_t is_aerodyn_input_passed_as_string =
            sim_controls_.is_aerodyn_input_path ? 0 : 1;  // reverse of is_aerodyn_input_path
        int32_t is_inflowwind_input_passed_as_string =
            sim_controls_.is_inflowwind_input_path ? 0 : 1;  // reverse of is_inflowwind_input_path
        int32_t store_HH_wind_speed_int = sim_controls_.store_HH_wind_speed ? 1 : 0;
        int32_t write_vtk_int = vtk_settings_.write_vtk ? 1 : 0;

        // Primary input file will be passed as path to the file
        char* aerodyn_input_pointer{sim_controls_.aerodyn_input.data()};
        int32_t aerodyn_input_length = static_cast<int32_t>(sim_controls_.aerodyn_input.size());

        char* inflowwind_input_pointer{sim_controls_.inflowwind_input.data()};
        int32_t inflowwind_input_length =
            static_cast<int32_t>(sim_controls_.inflowwind_input.size());

        ADI_C_Init(
            &is_aerodyn_input_passed_as_string,           // input: AD input is passed
            &aerodyn_input_pointer,                       // input: AD input file as string
            &aerodyn_input_length,                        // input: AD input file string length
            &is_inflowwind_input_passed_as_string,        // input: IfW input is passed
            &inflowwind_input_pointer,                    // input: IfW input file as string
            &inflowwind_input_length,                     // input: IfW input file string length
            sim_controls_.output_root_name.c_str(),       // input: rootname for ADI file writing
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
            &store_HH_wind_speed_int,                     // input: store HH wind speed
            &write_vtk_int,                               // input: write VTK output
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
     * @brief Set up the rotor motion for the current simulation step
     *
     * @details Updates the motion data (position, orientation, velocity, acceleration) for
     * the hub, nacelle, blade roots, and blade nodes of each turbine.
     */
    void SetupRotorMotion() {
        auto
            ADI_C_SetRotorMotion =
                lib_
                    .get_function<
                        void(int*, const float*, const double*, const float*, const float*, const float*, const double*, const float*, const float*, const float*, const double*, const float*, const float*, const int*, const float*, const double*, const float*, const float*, int*, char*)>(
                        "ADI_C_SetRotorMotion"
                    );

        // Loop through turbines and set the rotor motion
        int32_t turbine_number{0};
        for (const auto& td : turbines_) {
            // Turbine number is 1 indexed i.e. 1, 2, 3, ...
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
                &error_handling_.error_status,               // output: error status
                error_handling_.error_message.data()         // output: error message buffer
            );

            error_handling_.CheckError();
        }
    }

    /**
     * @brief Gets the aerodynamic loads on the rotor
     *
     * @details Fetches the current aerodynamic forces and moments acting on each blade node
     * for all turbines in the simulation.
     */
    void GetRotorAerodynamicLoads() {
        auto ADI_C_GetRotorLoads =
            lib_.get_function<void(int*, int*, float*, int*, char*)>("ADI_C_GetRotorLoads");

        // Loop through turbines and get the rotor loads
        int32_t turbine_number{0};
        for (auto& td : turbines_) {
            // Turbine number is 1 indexed i.e. 1, 2, 3, ...
            ++turbine_number;

            ADI_C_GetRotorLoads(
                &turbine_number,                      // input: current turbine number
                &td.blade_nodes.n_mesh_points,        // input: number of mesh points
                td.blade_nodes.loads.data()->data(),  // output: mesh force/moment array
                &error_handling_.error_status,        // output: error status
                error_handling_.error_message.data()  // output: error message buffer
            );

            error_handling_.CheckError();
        }
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
        is_initialized_ = false;
    }

private:
    bool is_initialized_{false};              //< Flag to check if the library is initialized
    util::dylib lib_;                         //< Dynamic library object for AeroDyn Inflow
    ErrorHandling error_handling_;            //< Error handling settings
    FluidProperties air_;                     //< Properties of the working fluid (air)
    EnvironmentalConditions env_conditions_;  //< Environmental conditions
    SimulationControls sim_controls_;         //< Simulation control settings
    VTKSettings vtk_settings_;                //< VTK output settings
    std::vector<TurbineData> turbines_;       //< Turbine data (1 per turbine)
};

}  // namespace openturbine::util
