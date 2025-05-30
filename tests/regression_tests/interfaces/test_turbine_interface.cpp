#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "interfaces/turbine/turbine_interface_builder.hpp"
#include "regression/test_utilities.hpp"

namespace openturbine::tests {

TEST(TurbineInterfaceTest, IEA15) {
    // Model parameters
    const auto n_blades{3};        // Number of blades in turbine
    const auto n_blade_nodes{11};  // Number of nodes per blade
    // const auto n_tower_nodes{11};  // Number of nodes in tower

    // Read WindIO yaml file
    const YAML::Node wio = YAML::LoadFile("interfaces_test_files/IEA-15-240-RWT.yaml");

    // Create interface builder
    auto builder = interfaces::TurbineInterfaceBuilder{};

    // Set solution parameters
    const double time_step{0.01};
    builder.Solution()
        .EnableDynamicSolve()
        .SetTimeStep(time_step)
        .SetDampingFactor(0.0)
        .SetMaximumNonlinearIterations(6)
        .SetAbsoluteErrorTolerance(1e-6)
        .SetRelativeErrorTolerance(1e-4)
        .SetOutputFile("TurbineInterfaceTest.IEA15");

    // Get turbine builder
    auto& turbine_builder = builder.Turbine();

    // Get the WindIO blade input
    const auto& wio_blade = wio["components"]["blade"];

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

    // Build turbine interface
    auto interface = builder.Build();

    // Loop through solution iterations
    for (auto i = 1U; i < 20U; ++i) {
        // Calculate time
        // const auto t{static_cast<double>(i) * time_step};

        // Take step
        const auto converged = interface.Step();

        // Check convergence
        ASSERT_EQ(converged, true);
    }
}

}  // namespace openturbine::tests
