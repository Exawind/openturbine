#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "elements/beams/hollow_circle_properties.hpp"
#include "interfaces/turbine/turbine_interface_builder.hpp"
#include "regression/test_utilities.hpp"

namespace openturbine::tests {

// This test builds the IEA-15-240-RWT turbine structure from WindIO yaml file
// and applies a tower load, generator torque, blade pitch, and yaw angle to test the
// structure's response.
TEST(TurbineInterfaceTest, IEA15_Structure) {
    const auto duration{0.1};        // Simulation duration in seconds
    const auto time_step{0.01};      // Time step for the simulation
    const auto n_blades{3};          // Number of blades in turbine
    const auto n_blade_nodes{11};    // Number of nodes per blade
    const auto n_tower_nodes{11};    // Number of nodes in tower
    const auto write_output{false};  // Write output file

    // Create interface builder
    auto builder = interfaces::TurbineInterfaceBuilder{};

    // Set solution parameters
    builder.Solution()
        .EnableDynamicSolve()
        .SetTimeStep(time_step)
        .SetDampingFactor(0.0)
        .SetGravity({0., 0., -9.81})
        .SetMaximumNonlinearIterations(6)
        .SetAbsoluteErrorTolerance(1e-6)
        .SetRelativeErrorTolerance(1e-4);

    if (write_output) {
        builder.Solution().SetOutputFile("TurbineInterfaceTest.IEA15");
    }

    // Read WindIO yaml file
    const YAML::Node wio = YAML::LoadFile("interfaces_test_files/IEA-15-240-RWT.yaml");

    // WindIO components
    const auto& wio_tower = wio["components"]["tower"];
    const auto& wio_nacelle = wio["components"]["nacelle"];
    const auto& wio_blade = wio["components"]["blade"];
    const auto& wio_hub = wio["components"]["hub"];

    //--------------------------------------------------------------------------
    // Build Turbine
    //--------------------------------------------------------------------------

    // Get turbine builder
    auto& turbine_builder = builder.Turbine();
    turbine_builder.SetAzimuthAngle(0.)
        .SetRotorApexToHub(0.)
        .SetHubDiameter(wio_hub["diameter"].as<double>())
        .SetConeAngle(wio_hub["cone_angle"].as<double>())
        .SetShaftTiltAngle(wio_nacelle["drivetrain"]["uptilt"].as<double>())
        .SetTowerAxisToRotorApex(wio_nacelle["drivetrain"]["overhang"].as<double>())
        .SetTowerTopToRotorApex(wio_nacelle["drivetrain"]["distance_tt_hub"].as<double>());

    //--------------------------------------------------------------------------
    // Build Blades
    //--------------------------------------------------------------------------

    // Loop through blades and set parameters
    for (auto j = 0U; j < n_blades; ++j) {
        // Get the blade builder
        auto& blade_builder = turbine_builder.Blade(j);

        // Set blade parameters
        blade_builder.SetElementOrder(n_blade_nodes - 1).PrescribedRootMotion(false);

        // Add reference axis coordinates (WindIO uses Z-axis as reference axis)
        const auto ref_axis = wio_blade["outer_shape_bem"]["reference_axis"];
        const auto axis_grid = ref_axis["x"]["grid"].as<std::vector<double>>();
        const auto x_values = ref_axis["x"]["values"].as<std::vector<double>>();
        const auto y_values = ref_axis["y"]["values"].as<std::vector<double>>();
        const auto z_values = ref_axis["z"]["values"].as<std::vector<double>>();
        for (auto i = 0U; i < axis_grid.size(); ++i) {
            blade_builder.AddRefAxisPoint(
                axis_grid[i], {x_values[i], y_values[i], z_values[i]},
                interfaces::components::ReferenceAxisOrientation::Z
            );
        }

        // Add reference axis twist
        const auto twist = wio_blade["outer_shape_bem"]["twist"];
        const auto twist_grid = twist["grid"].as<std::vector<double>>();
        const auto twist_values = twist["values"].as<std::vector<double>>();
        for (auto i = 0U; i < twist_grid.size(); ++i) {
            blade_builder.AddRefAxisTwist(twist_grid[i], twist_values[i]);
        }

        // Add blade section properties
        const auto stiff_matrix = wio_blade["elastic_properties_mb"]["six_x_six"]["stiff_matrix"];
        const auto inertia_matrix =
            wio_blade["elastic_properties_mb"]["six_x_six"]["inertia_matrix"];
        const auto k_grid = stiff_matrix["grid"].as<std::vector<double>>();
        const auto m_grid = inertia_matrix["grid"].as<std::vector<double>>();
        const auto n_sections = k_grid.size();
        if (m_grid.size() != k_grid.size()) {
            throw std::runtime_error("stiffness and mass matrices not on same grid");
        }
        for (auto i = 0U; i < n_sections; ++i) {
            if (abs(m_grid[i] - k_grid[i]) > 1e-8) {
                throw std::runtime_error("stiffness and mass matrices not on same grid");
            }
            const auto m = inertia_matrix["values"][i].as<std::vector<double>>();
            const auto k = stiff_matrix["values"][i].as<std::vector<double>>();
            blade_builder.AddSection(
                m_grid[i],
                {{
                    {m[0], m[1], m[2], m[3], m[4], m[5]},
                    {m[1], m[6], m[7], m[8], m[9], m[10]},
                    {m[2], m[7], m[11], m[12], m[13], m[14]},
                    {m[3], m[8], m[12], m[15], m[16], m[17]},
                    {m[4], m[9], m[13], m[16], m[18], m[19]},
                    {m[5], m[10], m[14], m[17], m[19], m[20]},
                }},
                {{
                    {k[0], k[1], k[2], k[3], k[4], k[5]},
                    {k[1], k[6], k[7], k[8], k[9], k[10]},
                    {k[2], k[7], k[11], k[12], k[13], k[14]},
                    {k[3], k[8], k[12], k[15], k[16], k[17]},
                    {k[4], k[9], k[13], k[16], k[18], k[19]},
                    {k[5], k[10], k[14], k[17], k[19], k[20]},
                }},
                interfaces::components::ReferenceAxisOrientation::Z
            );
        }
    }

    //--------------------------------------------------------------------------
    // Build Tower
    //--------------------------------------------------------------------------

    // Get the tower builder
    auto& tower_builder = turbine_builder.Tower();

    // Set tower parameters
    tower_builder
        .SetElementOrder(n_tower_nodes - 1)  // Set element order to num nodes -1
        .PrescribedRootMotion(false);        // Fix displacement of tower base node

    // Add reference axis coordinates (WindIO uses Z-axis as reference axis)
    const auto t_ref_axis = wio_tower["outer_shape_bem"]["reference_axis"];
    const auto axis_grid = t_ref_axis["x"]["grid"].as<std::vector<double>>();
    const auto x_values = t_ref_axis["x"]["values"].as<std::vector<double>>();
    const auto y_values = t_ref_axis["y"]["values"].as<std::vector<double>>();
    const auto z_values = t_ref_axis["z"]["values"].as<std::vector<double>>();
    for (auto i = 0U; i < axis_grid.size(); ++i) {
        tower_builder.AddRefAxisPoint(
            axis_grid[i], {x_values[i], y_values[i], z_values[i]},
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    // Add reference axis twist (zero for tower)
    tower_builder.AddRefAxisTwist(0.0, 0.0).AddRefAxisTwist(1.0, 0.0);

    // Find the tower material properties
    const auto t_layer = wio_tower["internal_structure_2d_fem"]["layers"][0];
    const auto t_material_name = t_layer["material"].as<std::string>();
    YAML::Node t_material;
    for (const auto& m : wio["materials"]) {
        if (m["name"] && m["name"].as<std::string>() == t_material_name) {
            t_material = m.as<YAML::Node>();
            break;
        }
    }
    if (!t_material) {
        throw std::runtime_error(
            "Material '" + t_material_name + "' not found in materials section"
        );
    }

    // Add tower section properties
    const auto t_diameter = wio_tower["outer_shape_bem"]["outer_diameter"];
    const auto t_diameter_grid = t_diameter["grid"].as<std::vector<double>>();
    const auto t_diameter_values = t_diameter["values"].as<std::vector<double>>();
    const auto t_wall_thickness = t_layer["thickness"]["values"].as<std::vector<double>>();
    for (auto i = 0U; i < t_diameter_grid.size(); ++i) {
        // Create section mass and stiffness matrices
        const auto section = GenerateHollowCircleSection(
            t_diameter_grid[i], t_material["E"].as<double>(), t_material["G"].as<double>(),
            t_material["rho"].as<double>(), t_diameter_values[i], t_wall_thickness[i],
            t_material["nu"].as<double>()
        );

        // Add section
        tower_builder.AddSection(
            t_diameter_grid[i], section.M_star, section.C_star,
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    //--------------------------------------------------------------------------
    // Interface
    //--------------------------------------------------------------------------

    // Build turbine interface
    auto interface = builder.Build();

    //--------------------------------------------------------------------------
    // Simulation
    //--------------------------------------------------------------------------

    // Apply load to tower-top node
    interface.Turbine().tower.nodes.back().loads = {1e5, 0., 0., 0., 0., 0.};

    // Apply torque to turbine shaft
    interface.Turbine().torque_control = 1e8;

    // Calculate number of steps
    const auto n_steps{static_cast<size_t>(duration / time_step)};

    // Loop through solution iterations
    for (auto i = 1U; i < n_steps; ++i) {
        // Calculate time
        const auto t{static_cast<double>(i) * time_step};

        // Set the pitch on blade 3
        interface.Turbine().blade_pitch_control[2] = t * 0.5;

        // Set the yaw angle
        interface.Turbine().yaw_control = t * 0.3;

        // Turn off the torque control after 500 steps
        if (i % 500 == 0) {
            interface.Turbine().torque_control = 0.;
        }

        // Take step
        const auto converged = interface.Step();

        // Check convergence
        ASSERT_EQ(converged, true);
    }

    // Check tower top position and orientation
    const auto& tower_top_node = interface.Turbine().tower.nodes.back();
    EXPECT_NEAR(tower_top_node.position[0], -0.00013297093554177921, 1e-10);
    EXPECT_NEAR(tower_top_node.position[1], 0.079678680774119062, 1e-10);
    EXPECT_NEAR(tower_top_node.position[2], 144.3794999717642, 1e-10);
    EXPECT_NEAR(tower_top_node.position[3], 0.70671615596672743, 1e-10);
    EXPECT_NEAR(tower_top_node.position[4], -0.0073063262326534895, 1e-10);
    EXPECT_NEAR(tower_top_node.position[5], -0.70744214100440284, 1e-10);
    EXPECT_NEAR(tower_top_node.position[6], -0.0049507195134699481, 1e-10);
}

}  // namespace openturbine::tests
