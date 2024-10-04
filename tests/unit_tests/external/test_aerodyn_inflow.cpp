
#include <gtest/gtest.h>

#include "src/utilities/aerodynamics/aerodyn_inflow.hpp"
#include "tests/unit_tests/regression/test_utilities.hpp"

namespace openturbine::tests {

TEST(AerodynInflowTest, ErrorHandling_NoThrow) {
    util::ErrorHandling error_handling;
    EXPECT_NO_THROW(error_handling.CheckError());
}

TEST(AerodynInflowTest, ErrorHandling_Throw) {
    util::ErrorHandling error_handling;
    error_handling.error_status = 1;  // Set error status to 1 to trigger an error
    EXPECT_THROW(error_handling.CheckError(), std::runtime_error);
}

TEST(AerodynInflowTest, FluidProperties_Default) {
    util::FluidProperties fluid_properties;
    EXPECT_NEAR(fluid_properties.density, 1.225f, 1e-6f);
    EXPECT_NEAR(fluid_properties.kinematic_viscosity, 1.464E-5f, 1e-6f);
    EXPECT_NEAR(fluid_properties.sound_speed, 335.f, 1e-6f);
    EXPECT_NEAR(fluid_properties.vapor_pressure, 1700.f, 1e-6f);
}

TEST(AerodynInflowTest, FluidProperties_Set) {
    util::FluidProperties fluid_properties;
    fluid_properties.density = 1.1f;
    fluid_properties.kinematic_viscosity = 1.5E-5f;
    fluid_properties.sound_speed = 340.f;
    fluid_properties.vapor_pressure = 1800.f;

    EXPECT_NEAR(fluid_properties.density, 1.1f, 1e-6f);
    EXPECT_NEAR(fluid_properties.kinematic_viscosity, 1.5E-5f, 1e-6f);
    EXPECT_NEAR(fluid_properties.sound_speed, 340.f, 1e-6f);
    EXPECT_NEAR(fluid_properties.vapor_pressure, 1800.f, 1e-6f);
}

TEST(AerodynInflowTest, EnvironmentalConditions_Default) {
    util::EnvironmentalConditions environmental_conditions;
    EXPECT_NEAR(environmental_conditions.gravity, 9.80665f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.atm_pressure, 103500.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.water_depth, 0.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.msl_offset, 0.f, 1e-6f);
}

TEST(AerodynInflowTest, EnvironmentalConditions_Set) {
    util::EnvironmentalConditions environmental_conditions;
    environmental_conditions.gravity = 9.79665f;
    environmental_conditions.atm_pressure = 103000.f;
    environmental_conditions.water_depth = 100.f;
    environmental_conditions.msl_offset = 10.f;

    EXPECT_NEAR(environmental_conditions.gravity, 9.79665f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.atm_pressure, 103000.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.water_depth, 100.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.msl_offset, 10.f, 1e-6f);
}

TEST(AerodynInflowTest, BladeInitialState_Constructor) {
    Array_7 root{1., 2., 3., 1., 0., 0., 0.};
    std::vector<Array_7> nodes{{4., 5., 6., 1., 0., 0., 0.}, {7., 8., 9., 1., 0., 0., 0.}};
    util::TurbineConfig::BladeInitialState blade_state{root, nodes};

    EXPECT_EQ(blade_state.root_initial_position, root);
    EXPECT_EQ(blade_state.node_initial_positions, nodes);
}

TEST(AerodynInflowTest, TurbineConfig_Constructor) {
    bool is_hawt{true};
    std::array<float, 3> ref_pos{10.f, 20.f, 30.f};
    Array_7 hub_pos{1., 2., 3., 1., 0., 0., 0.};
    Array_7 nacelle_pos{4., 5., 6., 1., 0., 0., 0.};

    std::vector<util::TurbineConfig::BladeInitialState> blade_states;
    for (int i = 0; i < 3; ++i) {
        Array_7 root = {static_cast<double>(i), 0., 0., 1., 0., 0., 0.};
        std::vector<Array_7> nodes = {
            {static_cast<double>(i), 1., 0., 1., 0., 0., 0.},
            {static_cast<double>(i), 2., 0., 1., 0., 0., 0.}};
        blade_states.emplace_back(root, nodes);
    }

    util::TurbineConfig turbine_config{is_hawt, ref_pos, hub_pos, nacelle_pos, blade_states};

    EXPECT_EQ(turbine_config.is_horizontal_axis, is_hawt);
    EXPECT_EQ(turbine_config.reference_position, ref_pos);
    EXPECT_EQ(turbine_config.hub_initial_position, hub_pos);
    EXPECT_EQ(turbine_config.nacelle_initial_position, nacelle_pos);
    EXPECT_EQ(turbine_config.blade_initial_states.size(), 3);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(
            turbine_config.blade_initial_states[i].root_initial_position[0], static_cast<double>(i)
        );
        EXPECT_EQ(turbine_config.blade_initial_states[i].node_initial_positions.size(), 2);
        EXPECT_EQ(
            turbine_config.blade_initial_states[i].node_initial_positions[0][0],
            static_cast<double>(i)
        );
        EXPECT_EQ(
            turbine_config.blade_initial_states[i].node_initial_positions[1][0],
            static_cast<double>(i)
        );
    }
}

TEST(AerodynInflowTest, TurbineConfig_Validate_InvalidConfiguration) {
    // Invalid configuration: No blades
    bool is_hawt{true};
    std::array<float, 3> ref_pos{10.f, 20.f, 30.f};
    Array_7 hub_pos{1., 2., 3., 1., 0., 0., 0.};
    Array_7 nacelle_pos{4., 5., 6., 1., 0., 0., 0.};
    std::vector<util::TurbineConfig::BladeInitialState> empty_blade_states;

    EXPECT_THROW(
        util::TurbineConfig(is_hawt, ref_pos, hub_pos, nacelle_pos, empty_blade_states),
        std::runtime_error
    );
}

/// Check if members of the provided array is equal to the provided expected array
template <typename T, size_t N>
void ExpectArrayNear(
    const std::array<T, N>& actual, const std::array<T, N>& expected,
    T epsilon = static_cast<T>(1e-6)
) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(actual[i], expected[i], epsilon) << "Element mismatch at index " << i;
    }
}

TEST(AerodynInflowTest, SetPositionAndOrientation) {
    std::array<double, 7> data = {1., 2., 3., 0.707107, 0.707107, 0., 0.};
    std::array<float, 3> position;
    std::array<double, 9> orientation;

    util::SetPositionAndOrientation(data, position, orientation);

    ExpectArrayNear(position, {1.f, 2.f, 3.f});
    ExpectArrayNear(orientation, {1., 0., 0., 0., 0., -1., 0., 1., 0.});
}

TEST(AerodynInflowTest, MeshData_Constructor_NumberOfNodes) {
    util::MeshData mesh_motion_data{1};
    EXPECT_EQ(mesh_motion_data.n_mesh_points, 1);
    EXPECT_EQ(mesh_motion_data.position.size(), 1);
    EXPECT_EQ(mesh_motion_data.orientation.size(), 1);
    EXPECT_EQ(mesh_motion_data.velocity.size(), 1);
    EXPECT_EQ(mesh_motion_data.acceleration.size(), 1);
    EXPECT_EQ(mesh_motion_data.loads.size(), 1);
}

TEST(AerodynInflowTest, MeshData_Constructor_Data) {
    size_t n_mesh_points{1};
    std::vector<std::array<double, 7>> mesh_data{{1., 2., 3., 0.707107, 0.707107, 0., 0.}};
    std::vector<std::array<float, 6>> mesh_velocities{{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}};
    std::vector<std::array<float, 6>> mesh_accelerations{{7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
    std::vector<std::array<float, 6>> mesh_loads = {{13.f, 14.f, 15.f, 16.f, 17.f, 18.f}};
    util::MeshData mesh_motion_data(
        n_mesh_points, mesh_data, mesh_velocities, mesh_accelerations, mesh_loads
    );

    EXPECT_EQ(mesh_motion_data.n_mesh_points, n_mesh_points);
    EXPECT_EQ(mesh_motion_data.position.size(), n_mesh_points);
    EXPECT_EQ(mesh_motion_data.orientation.size(), n_mesh_points);
    EXPECT_EQ(mesh_motion_data.velocity.size(), n_mesh_points);
    EXPECT_EQ(mesh_motion_data.acceleration.size(), n_mesh_points);
    EXPECT_EQ(mesh_motion_data.loads.size(), n_mesh_points);
    ExpectArrayNear(mesh_motion_data.position[0], {1.f, 2.f, 3.f});
    ExpectArrayNear(mesh_motion_data.orientation[0], {1., 0., 0., 0., 0., -1., 0., 1., 0.});
    ExpectArrayNear(mesh_motion_data.velocity[0], {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    ExpectArrayNear(mesh_motion_data.acceleration[0], {7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    ExpectArrayNear(mesh_motion_data.loads[0], {13.f, 14.f, 15.f, 16.f, 17.f, 18.f});
}

class MeshDataValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a mesh data with 2 mesh points
        mesh_data = std::make_unique<util::MeshData>(2);
        mesh_data->position = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
        mesh_data->orientation = {
            {1., 0., 0., 0., 1., 0., 0., 0., 1.}, {0., 1., 0., -1., 0., 0., 0., 0., 1.}};
        mesh_data->velocity = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
        mesh_data->acceleration = {
            {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
        mesh_data->loads = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
    }

    std::unique_ptr<util::MeshData> mesh_data;
};

TEST_F(MeshDataValidationTest, InvalidNumberOfMeshPoints) {
    mesh_data->n_mesh_points = 0;  // Should be at least 1
    EXPECT_THROW(mesh_data->Validate(), std::invalid_argument);
}

TEST_F(MeshDataValidationTest, MismatchedPositionSize) {
    mesh_data->position.pop_back();
    EXPECT_THROW(mesh_data->Validate(), std::invalid_argument);
}

TEST_F(MeshDataValidationTest, MismatchedOrientationSize) {
    mesh_data->orientation.pop_back();
    EXPECT_THROW(mesh_data->Validate(), std::invalid_argument);
}

TEST_F(MeshDataValidationTest, MismatchedVelocitySize) {
    mesh_data->velocity.pop_back();
    EXPECT_THROW(mesh_data->Validate(), std::invalid_argument);
}

TEST_F(MeshDataValidationTest, MismatchedAccelerationSize) {
    mesh_data->acceleration.pop_back();
    EXPECT_THROW(mesh_data->Validate(), std::invalid_argument);
}

TEST_F(MeshDataValidationTest, MismatchedLoadsSize) {
    mesh_data->loads.pop_back();
    EXPECT_THROW(mesh_data->Validate(), std::invalid_argument);
}

TEST(AerodynInflowTest, TurbineData_Constructor) {
    // Set up 3 blades with 2 nodes each
    std::vector<util::TurbineConfig::BladeInitialState> blade_states;
    for (size_t i = 0; i < 3; ++i) {
        util::TurbineConfig::BladeInitialState blade_state(
            {0., 0., 90., 1., 0., 0., 0.},  // root_initial_position
            {
                {0., 5., 90., 1., 0., 0., 0.},  // node_initial_positions - 1
                {0., 10., 90., 1., 0., 0., 0.}  // node_initial_positions - 2
            }

        );
        blade_states.push_back(blade_state);
    }
    util::TurbineConfig tc(
        true,                           // is_horizontal_axis
        {0.f, 0.f, 0.f},                // reference_position
        {0., 0., 90., 1., 0., 0., 0.},  // hub_initial_position
        {0., 0., 90., 1., 0., 0., 0.},  // nacelle_initial_position
        blade_states
    );

    util::TurbineData turbine_data(tc);

    // Check basic properties
    EXPECT_EQ(turbine_data.n_blades, 3);
    EXPECT_EQ(turbine_data.hub.n_mesh_points, 1);
    EXPECT_EQ(turbine_data.nacelle.n_mesh_points, 1);
    EXPECT_EQ(turbine_data.blade_roots.n_mesh_points, 3);
    EXPECT_EQ(turbine_data.blade_nodes.n_mesh_points, 6);  // 3 blades * 2 nodes each

    // Check hub and nacelle positions
    ExpectArrayNear(turbine_data.hub.position[0], {0.f, 0.f, 90.f});
    ExpectArrayNear(turbine_data.nacelle.position[0], {0.f, 0.f, 90.f});

    // Check blade roots
    for (size_t i = 0; i < 3; ++i) {
        ExpectArrayNear(turbine_data.blade_roots.position[i], {0.f, 0.f, 90.f});
    }

    // Check blade nodes
    for (size_t i = 0; i < 6; ++i) {
        float expected_y = (i % 2 == 0) ? 5.f : 10.f;
        ExpectArrayNear(turbine_data.blade_nodes.position[i], {0.f, expected_y, 90.f});
    }

    // Check blade_nodes_to_blade_num_mapping
    std::vector<int32_t> expected_blade_nodes_to_blade_num_mapping = {1, 1, 2, 2, 3, 3};
    ASSERT_EQ(turbine_data.blade_nodes_to_blade_num_mapping.size(), 6);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(
            turbine_data.blade_nodes_to_blade_num_mapping[i],
            expected_blade_nodes_to_blade_num_mapping[i]
        ) << "Mismatch at index "
          << i;
    }

    // Check node_indices_by_blade
    ASSERT_EQ(turbine_data.node_indices_by_blade.size(), 3);
    std::vector<std::vector<size_t>> expected_node_indices_by_blade = {{0, 1}, {2, 3}, {4, 5}};
    for (size_t blade = 0; blade < 3; ++blade) {
        ASSERT_EQ(turbine_data.node_indices_by_blade[blade].size(), 2)
            << "Incorrect number of nodes for blade " << blade;
        EXPECT_EQ(turbine_data.node_indices_by_blade[blade], expected_node_indices_by_blade[blade])
            << "Incorrect node indices for blade " << blade;
    }

    // Additional check: Verify that node_indices_by_blade correctly maps to
    // blade_nodes_to_blade_num_mapping
    for (size_t blade = 0; blade < 3; ++blade) {
        for (size_t node : turbine_data.node_indices_by_blade[blade]) {
            EXPECT_EQ(turbine_data.blade_nodes_to_blade_num_mapping[node], blade + 1)
                << "Mismatch between node_indices_by_blade and blade_nodes_to_blade_num_mapping for "
                   "blade "
                << blade;
        }
    }
}

class TurbineDataValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        blade_states.clear();
        // Set up 3 blades with 2 nodes each
        for (size_t i = 0; i < 3; ++i) {
            util::TurbineConfig::BladeInitialState blade_state(
                {0., 0., 90., 1., 0., 0., 0.},  // root_initial_position
                {
                    {0., 5., 90., 1., 0., 0., 0.},  // node_initial_positions - 1
                    {0., 10., 90., 1., 0., 0., 0.}  // node_initial_positions - 2
                }
            );
            blade_states.push_back(blade_state);
        }
        tc = std::make_unique<util::TurbineConfig>(
            true,                                                // is_horizontal_axis
            std::array<float, 3>{0.f, 0.f, 0.f},                 // reference_position
            std::array<double, 7>{0., 0., 90., 1., 0., 0., 0.},  // hub_initial_position
            std::array<double, 7>{0., 0., 90., 1., 0., 0., 0.},  // nacelle_initial_position
            blade_states
        );
        turbine_data = std::make_unique<util::TurbineData>(*tc);
    }

    std::vector<util::TurbineConfig::BladeInitialState> blade_states;
    std::unique_ptr<util::TurbineConfig> tc;
    std::unique_ptr<util::TurbineData> turbine_data;
};

TEST_F(TurbineDataValidationTest, InvalidNumberOfBlades) {
    turbine_data->n_blades = 0;  // Should be at least 1
    EXPECT_THROW(turbine_data->Validate(), std::runtime_error);
}

TEST_F(TurbineDataValidationTest, MismatchBladeRoots) {
    turbine_data->blade_roots.n_mesh_points = 2;  // Should be 3
    EXPECT_THROW(turbine_data->Validate(), std::runtime_error);
}

TEST_F(TurbineDataValidationTest, MismatchBladeNodeMapping) {
    turbine_data->blade_nodes_to_blade_num_mapping.pop_back();
    EXPECT_THROW(turbine_data->Validate(), std::runtime_error);
}

TEST_F(TurbineDataValidationTest, MismatchNodeIndices) {
    turbine_data->node_indices_by_blade.pop_back();
    EXPECT_THROW(turbine_data->Validate(), std::runtime_error);
}

TEST_F(TurbineDataValidationTest, MismatchTotalBladeNodes) {
    turbine_data->blade_nodes.n_mesh_points = 5;  // Should be 6
    EXPECT_THROW(turbine_data->Validate(), std::runtime_error);
}

TEST_F(TurbineDataValidationTest, InvalidHubMeshPoints) {
    turbine_data->hub.n_mesh_points = 2;  // Should be 1
    EXPECT_THROW(turbine_data->Validate(), std::runtime_error);
}

TEST_F(TurbineDataValidationTest, InvalidNacelleMeshPoints) {
    turbine_data->nacelle.n_mesh_points = 0;  // Should be 1
    EXPECT_THROW(turbine_data->Validate(), std::runtime_error);
}

TEST(AerodynInflowTest, SimulationControls_Default) {
    util::SimulationControls simulation_controls;
    EXPECT_EQ(simulation_controls.is_aerodyn_input_path, true);
    EXPECT_EQ(simulation_controls.is_inflowwind_input_path, true);
    EXPECT_EQ(simulation_controls.interpolation_order, 1);
    EXPECT_EQ(simulation_controls.time_step, 0.1);
    EXPECT_EQ(simulation_controls.max_time, 600.0);
    EXPECT_EQ(simulation_controls.total_elapsed_time, 0.0);
    EXPECT_EQ(simulation_controls.n_time_steps, 0);
    EXPECT_EQ(simulation_controls.store_HH_wind_speed, 1);
    EXPECT_EQ(simulation_controls.transpose_DCM, 1);
    EXPECT_EQ(simulation_controls.debug_level, 0);
    EXPECT_EQ(simulation_controls.output_format, 1);
    EXPECT_EQ(simulation_controls.output_time_step, 0.1);
    EXPECT_STREQ(simulation_controls.output_root_name.data(), "ADI_out");
    EXPECT_EQ(simulation_controls.n_channels, 0);
    EXPECT_STREQ(simulation_controls.channel_names_c.data(), "");
    EXPECT_STREQ(simulation_controls.channel_units_c.data(), "");
}

TEST(AerodynInflowTest, SimulationControls_Set) {
    util::SimulationControls simulation_controls;
    simulation_controls.is_aerodyn_input_path = false;
    simulation_controls.is_inflowwind_input_path = false;
    simulation_controls.interpolation_order = 2;
    simulation_controls.time_step = 0.2;
    simulation_controls.max_time = 1200.0;
    simulation_controls.total_elapsed_time = 100.0;
    simulation_controls.n_time_steps = 10;
    simulation_controls.store_HH_wind_speed = 0;
    simulation_controls.transpose_DCM = 0;
    simulation_controls.debug_level = 1;
    simulation_controls.output_format = 1;
    simulation_controls.output_time_step = 1.0;
    simulation_controls.n_channels = 1;
    std::strncpy(
        simulation_controls.channel_names_c.data(), "test_channel",
        simulation_controls.channel_names_c.size() - 1
    );
    simulation_controls.channel_names_c[simulation_controls.channel_names_c.size() - 1] = '\0';
    std::strncpy(
        simulation_controls.channel_units_c.data(), "test_unit",
        simulation_controls.channel_units_c.size() - 1
    );
    simulation_controls.channel_units_c[simulation_controls.channel_units_c.size() - 1] = '\0';

    EXPECT_EQ(simulation_controls.is_aerodyn_input_path, 0);
    EXPECT_EQ(simulation_controls.is_inflowwind_input_path, 0);
    EXPECT_EQ(simulation_controls.interpolation_order, 2);
    EXPECT_EQ(simulation_controls.time_step, 0.2);
    EXPECT_EQ(simulation_controls.max_time, 1200.0);
    EXPECT_EQ(simulation_controls.total_elapsed_time, 100.0);
    EXPECT_EQ(simulation_controls.n_time_steps, 10);
    EXPECT_EQ(simulation_controls.store_HH_wind_speed, 0);
    EXPECT_EQ(simulation_controls.transpose_DCM, 0);
    EXPECT_EQ(simulation_controls.debug_level, 1);
    EXPECT_EQ(simulation_controls.output_format, 1);
    EXPECT_EQ(simulation_controls.output_time_step, 1.0);
    EXPECT_STREQ(simulation_controls.output_root_name.data(), "ADI_out");
    EXPECT_EQ(simulation_controls.n_channels, 1);
    EXPECT_STREQ(simulation_controls.channel_names_c.data(), "test_channel");
    EXPECT_STREQ(simulation_controls.channel_units_c.data(), "test_unit");
}

TEST(AerodynInflowTest, VTKSettings_Default) {
    util::VTKSettings vtk_settings;
    EXPECT_EQ(vtk_settings.write_vtk, 0);
    EXPECT_EQ(vtk_settings.vtk_type, 1);
    ExpectArrayNear(vtk_settings.vtk_nacelle_dimensions, {-2.5f, -2.5f, 0.f, 10.f, 5.f, 5.f});
    EXPECT_EQ(vtk_settings.vtk_hub_radius, 1.5f);
}

TEST(AerodynInflowTest, VTKSettings_Set) {
    util::VTKSettings vtk_settings;
    vtk_settings.write_vtk = 1;
    vtk_settings.vtk_type = 2;
    vtk_settings.vtk_nacelle_dimensions = {-1.5f, -1.5f, 0.f, 5.f, 2.5f, 2.5f};
    vtk_settings.vtk_hub_radius = 1.0f;

    EXPECT_EQ(vtk_settings.write_vtk, 1);
    EXPECT_EQ(vtk_settings.vtk_type, 2);
    ExpectArrayNear(vtk_settings.vtk_nacelle_dimensions, {-1.5f, -1.5f, 0.f, 5.f, 2.5f, 2.5f});
    EXPECT_EQ(vtk_settings.vtk_hub_radius, 1.0f);
}

/// Helper function to get the shared library path
std::string GetSharedLibraryPath() {
    const std::filesystem::path project_root = FindProjectRoot();
    const std::filesystem::path full_path =
        project_root / "build/tests/unit_tests/aerodyn_inflow_c_binding.dll";
    return full_path.string();
}

TEST(AerodynInflowTest, AeroDynInflowLibrary_DefaultConstructor) {
    // Load the shared library
    const std::string path = GetSharedLibraryPath();
    util::AeroDynInflowLibrary aerodyn_inflow_library(path);

    // Check initial error handling state
    EXPECT_EQ(aerodyn_inflow_library.GetErrorHandling().error_status, 0);
    EXPECT_STREQ(aerodyn_inflow_library.GetErrorHandling().error_message.data(), "");

    // Check default values for other important members
    EXPECT_EQ(aerodyn_inflow_library.GetFluidProperties().density, 1.225f);
    EXPECT_EQ(aerodyn_inflow_library.GetEnvironmentalConditions().gravity, 9.80665f);
    EXPECT_EQ(aerodyn_inflow_library.GetSimulationControls().debug_level, 0);
    EXPECT_EQ(aerodyn_inflow_library.GetSimulationControls().transpose_DCM, 1);
    EXPECT_EQ(aerodyn_inflow_library.GetVTKSettings().write_vtk, 0);
}

/* TEST(AerodynInflowTest, AeroDynInflowLibrary_FullLoop) {
    // Set up simulation parameters
    util::SimulationControls sim_controls{
        .is_aerodyn_input_path = false,
        .is_inflowwind_input_path = false,
        .time_step = 0.0125,
        .max_time = 10.0,
        .interpolation_order = 2,
        .store_HH_wind_speed = false,
        .transpose_DCM = 1,
        .debug_level = 1,
        .output_format = 0,
        .output_time_step = 0.1};

    // Set up environmental conditions and fluid properties
    util::EnvironmentalConditions env_conditions{
        .gravity = 9.80665f, .atm_pressure = 103500.0f, .water_depth = 0.0f, .msl_offset = 0.0f};

    util::FluidProperties fluid_props{
        .density = 1.225f,
        .kinematic_viscosity = 1.464E-05f,
        .sound_speed = 335.0f,
        .vapor_pressure = 1700.0f};

    // Set up turbine settings
    std::array<double, 7> hub_data = {0.0, 0.0, 90.0, 1.0, 0.0, 0.0, 0.0};
    std::array<double, 7> nacelle_data = {0.0, 0.0, 90.0, 1.0, 0.0, 0.0, 0.0};
    std::vector<std::array<double, 7>> root_data(3, {0.0, 0.0, 90.0, 1.0, 0.0, 0.0, 0.0});

    util::TurbineSettings turbine_settings(hub_data, nacelle_data, root_data, 1, 3);

    // Set up VTK settings
    util::VTKSettings vtk_settings{};

    // Load the shared library and initialize AeroDynInflowLibrary
    const std::string path = GetSharedLibraryPath();
    util::AeroDynInflowLibrary aerodyn_inflow_library(
        path, util::ErrorHandling{}, fluid_props, env_conditions, turbine_settings,
        util::StructuralMesh{}, sim_controls, vtk_settings
    );

    // Pre-initialize and setup rotor
    EXPECT_NO_THROW(aerodyn_inflow_library.PreInitialize());
    EXPECT_NO_THROW(aerodyn_inflow_library.SetupRotor(1, true, {0.0f, 0.0f, 0.0f}));

    // Initialize with input files
    const std::filesystem::path project_root = FindProjectRoot();
    std::filesystem::path input_path = project_root / "tests/unit_tests/external/";
    std::vector<std::string> adiAD_input_string_array = {(input_path / "ad_primary.dat").string()};
    std::vector<std::string> adiIfW_input_string_array = {(input_path / "ifw_primary.dat").string()};

    EXPECT_NO_THROW(
        aerodyn_inflow_library.Initialize(adiAD_input_string_array, adiIfW_input_string_array)
    );

    // Set up motion data for hub, nacelle, root, and mesh
    util::MeshData hub_motion{1};
    util::MeshData nacelle_motion{1};
    util::MeshData root_motion{3};
    util::MeshData mesh_motion{1};

    // Set up rotor motion
    EXPECT_NO_THROW(aerodyn_inflow_library.SetupRotorMotion(
        1, hub_motion, nacelle_motion, root_motion, mesh_motion
    ));

    // Simulate for a few time steps
    double current_time = 0.0;
    double next_time = sim_controls.time_step;
    std::vector<float> output_channel_values;
    std::vector<std::array<float, 6>> mesh_force_moment(1);  // Assuming 1 mesh point

    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(aerodyn_inflow_library.UpdateStates(current_time, next_time));
        EXPECT_NO_THROW(
            aerodyn_inflow_library.CalculateOutputChannels(next_time, output_channel_values)
        );
        EXPECT_NO_THROW(aerodyn_inflow_library.GetRotorAerodynamicLoads(1, mesh_force_moment));

        current_time = next_time;
        next_time += sim_controls.time_step;
    }

    // End simulation
    EXPECT_NO_THROW(aerodyn_inflow_library.Finalize());
} */

}  // namespace openturbine::tests