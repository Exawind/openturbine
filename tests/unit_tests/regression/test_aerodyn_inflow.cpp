#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/utilities/aerodynamics/aerodyn_inflow.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::tests {

#ifdef OpenTurbine_BUILD_OPENFAST_ADI

TEST(AerodynInflowTest, ErrorHandling_NoThrow) {
    util::ErrorHandling error_handling;
    EXPECT_NO_THROW(error_handling.CheckError());
}

TEST(AerodynInflowTest, ErrorHandling_Throw) {
    util::ErrorHandling error_handling;
    error_handling.error_status = 1;
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

    ExpectArrayNear(position, {1.0f, 2.0f, 3.0f});
    ExpectArrayNear(orientation, {1., 0., 0., 0., 0., -1., 0., 1., 0.});
}

TEST(AerodynInflowTest, TurbineSettings_Default) {
    util::TurbineSettings turbine_settings;
    EXPECT_EQ(turbine_settings.n_turbines, 1);
    EXPECT_EQ(turbine_settings.n_blades, 3);
    ExpectArrayNear(turbine_settings.initial_hub_position, {0.f, 0.f, 0.f});
    ExpectArrayNear(turbine_settings.initial_hub_orientation, {1., 0., 0., 0., 1., 0., 0., 0., 1.});
    ExpectArrayNear(turbine_settings.initial_nacelle_position, {0.f, 0.f, 0.f});
    ExpectArrayNear(
        turbine_settings.initial_nacelle_orientation, {1., 0., 0., 0., 1., 0., 0., 0., 1.}
    );
    for (size_t i = 0; i < turbine_settings.initial_root_position.size(); ++i) {
        ExpectArrayNear(turbine_settings.initial_root_position[i], {0.f, 0.f, 0.f});
        ExpectArrayNear(
            turbine_settings.initial_root_orientation[i], {1., 0., 0., 0., 1., 0., 0., 0., 1.}
        );
    }
}

TEST(AerodynInflowTest, TurbineSettings_Set_1T1B) {
    int n_turbines{1};
    int n_blades{1};
    std::array<double, 7> hub_data = {1., 2., 3., 0.707107, 0.707107, 0., 0.};
    std::array<double, 7> nacelle_data = {4., 5., 6., 0.707107, 0.707107, 0., 0.};
    std::vector<std::array<double, 7>> root_data = {{7., 8., 9., 0.707107, 0.707107, 0., 0.}};

    util::TurbineSettings turbine_settings(hub_data, nacelle_data, root_data, n_turbines, n_blades);

    EXPECT_EQ(turbine_settings.n_turbines, 1);
    EXPECT_EQ(turbine_settings.n_blades, 1);
    ExpectArrayNear(turbine_settings.initial_hub_position, {1.f, 2.f, 3.f});
    ExpectArrayNear(turbine_settings.initial_hub_orientation, {1., 0., 0., 0., 0., -1., 0., 1., 0.});
    ExpectArrayNear(turbine_settings.initial_nacelle_position, {4.f, 5.f, 6.f});
    ExpectArrayNear(
        turbine_settings.initial_nacelle_orientation, {1., 0., 0., 0., 0., -1., 0., 1., 0.}
    );
    ExpectArrayNear(turbine_settings.initial_root_position[0], {7.f, 8.f, 9.f});
    ExpectArrayNear(
        turbine_settings.initial_root_orientation[0], {1., 0., 0., 0., 0., -1., 0., 1., 0.}
    );
}

TEST(AerodynInflowTest, StructuralMesh_Default) {
    util::StructuralMesh structural_mesh;
    EXPECT_EQ(structural_mesh.n_mesh_points, 1);
    EXPECT_EQ(structural_mesh.initial_mesh_position.size(), 1);
    EXPECT_EQ(structural_mesh.initial_mesh_orientation.size(), 1);
    EXPECT_EQ(structural_mesh.mesh_point_to_blade_num.size(), 1);
    ExpectArrayNear(structural_mesh.initial_mesh_position[0], {0.f, 0.f, 0.f});
    ExpectArrayNear(
        structural_mesh.initial_mesh_orientation[0], {1., 0., 0., 0., 1., 0., 0., 0., 1.}
    );
    EXPECT_EQ(structural_mesh.mesh_point_to_blade_num[0], 1);
}

TEST(AerodynInflowTest, StructuralMesh_Set) {
    std::vector<std::array<double, 7>> mesh_data = {{1., 2., 3., 0.707107, 0.707107, 0., 0.}};
    std::vector<int> mesh_point_to_blade_num = {1};
    util::StructuralMesh structural_mesh(mesh_data, mesh_point_to_blade_num, 1);

    EXPECT_EQ(structural_mesh.n_mesh_points, 1);
    EXPECT_EQ(structural_mesh.initial_mesh_position.size(), 1);
    EXPECT_EQ(structural_mesh.initial_mesh_orientation.size(), 1);
    EXPECT_EQ(structural_mesh.mesh_point_to_blade_num.size(), 1);
    ExpectArrayNear(structural_mesh.initial_mesh_position[0], {1.f, 2.f, 3.f});
    ExpectArrayNear(
        structural_mesh.initial_mesh_orientation[0], {1., 0., 0., 0., 0., -1., 0., 1., 0.}
    );
    EXPECT_EQ(structural_mesh.mesh_point_to_blade_num[0], 1);
}

TEST(AerodynInflowTest, MeshMotionData_Default) {
    util::MeshMotionData mesh_motion_data;
    EXPECT_EQ(mesh_motion_data.position.size(), 0);
    EXPECT_EQ(mesh_motion_data.orientation.size(), 0);
    EXPECT_EQ(mesh_motion_data.velocity.size(), 0);
    EXPECT_EQ(mesh_motion_data.acceleration.size(), 0);
}

TEST(AerodynInflowTest, MeshMotionData_Set) {
    std::vector<std::array<double, 7>> mesh_data = {{1., 2., 3., 0.707107, 0.707107, 0., 0.}};
    std::vector<std::array<float, 6>> mesh_velocities = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}};
    std::vector<std::array<float, 6>> mesh_accelerations = {{7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
    util::MeshMotionData mesh_motion_data(mesh_data, mesh_velocities, mesh_accelerations, 1);

    EXPECT_EQ(mesh_motion_data.position.size(), 1);
    EXPECT_EQ(mesh_motion_data.orientation.size(), 1);
    EXPECT_EQ(mesh_motion_data.velocity.size(), 1);
    EXPECT_EQ(mesh_motion_data.acceleration.size(), 1);
    ExpectArrayNear(mesh_motion_data.position[0], {1.f, 2.f, 3.f});
    ExpectArrayNear(mesh_motion_data.orientation[0], {1., 0., 0., 0., 0., -1., 0., 1., 0.});
    ExpectArrayNear(mesh_motion_data.velocity[0], {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    ExpectArrayNear(mesh_motion_data.acceleration[0], {7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
}

TEST(AerodynInflowTest, MeshMotionData_CheckArraySize_NoThrow) {
    std::vector<std::array<double, 7>> mesh_data = {{1., 2., 3., 0.707107, 0.707107, 0., 0.}};
    std::vector<std::array<float, 6>> mesh_velocities = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}};
    std::vector<std::array<float, 6>> mesh_accelerations = {{7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
    util::MeshMotionData mesh_motion_data(mesh_data, mesh_velocities, mesh_accelerations, 1);

    mesh_motion_data.CheckArraySize(mesh_motion_data.position, 1, 3, "position", "mesh motion data");
    mesh_motion_data.CheckArraySize(
        mesh_motion_data.orientation, 1, 9, "orientation", "mesh motion data"
    );
    mesh_motion_data.CheckArraySize(mesh_motion_data.velocity, 1, 6, "velocity", "mesh motion data");
    mesh_motion_data.CheckArraySize(
        mesh_motion_data.acceleration, 1, 6, "acceleration", "mesh motion data"
    );
}

TEST(AerodynInflowTest, MeshMotionData_CheckArraySize_ExpectThrow) {
    std::vector<std::array<double, 7>> mesh_data = {{1., 2., 3., 0.707107, 0.707107, 0., 0.}};
    std::vector<std::array<float, 6>> mesh_velocities = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}};
    std::vector<std::array<float, 6>> mesh_accelerations = {{7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
    util::MeshMotionData mesh_motion_data(mesh_data, mesh_velocities, mesh_accelerations, 1);

    EXPECT_THROW(
        mesh_motion_data.CheckArraySize(
            mesh_motion_data.position, 2, 3, "position", "mesh motion data"  // Expected 1 row
        ),
        std::invalid_argument
    );
}

TEST(AerodynInflowTest, SimulationControls_Default) {
    util::SimulationControls simulation_controls;
    EXPECT_EQ(simulation_controls.aerodyn_input_passed, 1);
    EXPECT_EQ(simulation_controls.inflowwind_input_passed, 1);
    EXPECT_EQ(simulation_controls.interpolation_order, 1);
    EXPECT_EQ(simulation_controls.time_step, 0.1);
    EXPECT_EQ(simulation_controls.max_time, 600.0);
    EXPECT_EQ(simulation_controls.total_elapsed_time, 0.0);
    EXPECT_EQ(simulation_controls.n_time_steps, 0);
    EXPECT_EQ(simulation_controls.store_HH_wind_speed, 1);
    EXPECT_EQ(simulation_controls.transpose_DCM, 1);
    EXPECT_EQ(simulation_controls.debug_level, 0);
    EXPECT_EQ(simulation_controls.output_format, 0);
    EXPECT_EQ(simulation_controls.output_time_step, 0.0);
    EXPECT_STREQ(simulation_controls.output_root_name.data(), "Output_ADIlib_default");
    EXPECT_EQ(simulation_controls.n_channels, 0);
    EXPECT_STREQ(simulation_controls.channel_names_c.data(), "");
    EXPECT_STREQ(simulation_controls.channel_units_c.data(), "");
}

TEST(AerodynInflowTest, SimulationControls_Set) {
    util::SimulationControls simulation_controls;
    simulation_controls.aerodyn_input_passed = 0;
    simulation_controls.inflowwind_input_passed = 0;
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
    std::strncpy(
        simulation_controls.output_root_name.data(), "Output_ADIlib_test",
        simulation_controls.output_root_name.size() - 1
    );
    simulation_controls.output_root_name[simulation_controls.output_root_name.size() - 1] = '\0';
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

    EXPECT_EQ(simulation_controls.aerodyn_input_passed, 0);
    EXPECT_EQ(simulation_controls.inflowwind_input_passed, 0);
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
    EXPECT_STREQ(simulation_controls.output_root_name.data(), "Output_ADIlib_test");
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
        project_root / "build/tests/unit_tests/libaerodyn_inflow_c_binding";

#ifdef __APPLE__
    return full_path.string() + ".dylib";
#elif __linux__
    return full_path.string() + ".so";
#else  // Windows
    return full_path.string() + ".dll";
#endif
}

TEST(AerodynInflowTest, AeroDynInflowLibrary_DefaultConstructor) {
    // Load the shared library
    const std::string path = GetSharedLibraryPath();
    util::AeroDynInflowLibrary aerodyn_inflow_library(path);

    // Check initial error handling state
    EXPECT_EQ(aerodyn_inflow_library.error_handling.error_status, 0);
    EXPECT_STREQ(aerodyn_inflow_library.error_handling.error_message.data(), "");

    // Check default values for other important members
    EXPECT_EQ(aerodyn_inflow_library.turbine_settings.n_turbines, 1);
    EXPECT_EQ(aerodyn_inflow_library.sim_controls.debug_level, 0);
    EXPECT_EQ(aerodyn_inflow_library.sim_controls.transpose_DCM, 1);
}

#endif

}  // namespace openturbine::tests