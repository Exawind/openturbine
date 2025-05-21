#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "interfaces/blade/blade_interface_builder.hpp"
#include "regression/test_utilities.hpp"

namespace openturbine::tests {

TEST(BladeInterfaceTest, BladeWindIO) {
    // Read WindIO yaml file
    const YAML::Node windio = YAML::LoadFile("interfaces_test_files/IEA-15-240-RWT.yaml");

    // Create interface builder
    auto builder = interfaces::BladeInterfaceBuilder{};

    // Set solution parameters
    const double time_step{0.01};
    builder.Solution()
        .EnableDynamicSolve()
        .SetTimeStep(time_step)
        .SetDampingFactor(0.0)
        .SetMaximumNonlinearIterations(6)
        .SetAbsoluteErrorTolerance(1e-6)
        .SetRelativeErrorTolerance(1e-4);

#ifdef OpenTurbine_ENABLE_VTK
    builder.Solution().SetVTKOutputPath("BladeInterfaceTest.BladeWindIO/blade_####");
#endif

    // Set blade parameters
    // Use prescribed root motion to fix root rotation
    const auto n_nodes{11};
    builder.Blade().SetElementOrder(n_nodes - 1).PrescribedRootMotion(true);

    // Add reference axis coordinates
    const auto blade_axis = windio["components"]["blade"]["reference_axis"];
    const auto ref_axis_n_coord_points = blade_axis["grid"].size();
    for (auto i = 0U; i < ref_axis_n_coord_points; ++i) {
        builder.Blade().AddRefAxisPoint(
            blade_axis["grid"][i].as<double>(),
            {
                blade_axis["x"][i].as<double>(),
                blade_axis["y"][i].as<double>(),
                blade_axis["z"][i].as<double>(),
            },
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    // Add reference axis twist
    const auto twist = windio["components"]["blade"]["outer_shape"]["twist"];
    const auto ref_axis_n_twist_points = twist["grid"].size();
    for (auto i = 0U; i < ref_axis_n_twist_points; ++i) {
        builder.Blade().AddRefAxisTwist(
            twist["grid"][i].as<double>(), twist["values"][i].as<double>()
        );
    }

    // Add blade section properties
    const auto stiffness_matrix =
        windio["components"]["blade"]["elastic_properties"]["stiffness_matrix"];
    const auto k_grid(stiffness_matrix["grid"].as<std::vector<double>>());
    const auto k11(stiffness_matrix["K11"].as<std::vector<double>>());
    const auto k12(stiffness_matrix["K12"].as<std::vector<double>>());
    const auto k13(stiffness_matrix["K13"].as<std::vector<double>>());
    const auto k14(stiffness_matrix["K14"].as<std::vector<double>>());
    const auto k15(stiffness_matrix["K15"].as<std::vector<double>>());
    const auto k16(stiffness_matrix["K16"].as<std::vector<double>>());
    const auto k22(stiffness_matrix["K22"].as<std::vector<double>>());
    const auto k23(stiffness_matrix["K23"].as<std::vector<double>>());
    const auto k24(stiffness_matrix["K24"].as<std::vector<double>>());
    const auto k25(stiffness_matrix["K25"].as<std::vector<double>>());
    const auto k26(stiffness_matrix["K26"].as<std::vector<double>>());
    const auto k33(stiffness_matrix["K33"].as<std::vector<double>>());
    const auto k34(stiffness_matrix["K34"].as<std::vector<double>>());
    const auto k35(stiffness_matrix["K35"].as<std::vector<double>>());
    const auto k36(stiffness_matrix["K36"].as<std::vector<double>>());
    const auto k44(stiffness_matrix["K44"].as<std::vector<double>>());
    const auto k45(stiffness_matrix["K45"].as<std::vector<double>>());
    const auto k46(stiffness_matrix["K46"].as<std::vector<double>>());
    const auto k55(stiffness_matrix["K55"].as<std::vector<double>>());
    const auto k56(stiffness_matrix["K56"].as<std::vector<double>>());
    const auto k66(stiffness_matrix["K66"].as<std::vector<double>>());

    const auto inertia_matrix =
        windio["components"]["blade"]["elastic_properties"]["inertia_matrix"];
    const auto m_grid(inertia_matrix["grid"].as<std::vector<double>>());
    const auto mass(inertia_matrix["mass"].as<std::vector<double>>());
    const auto cm_x(inertia_matrix["cm_x"].as<std::vector<double>>());
    const auto cm_y(inertia_matrix["cm_y"].as<std::vector<double>>());
    const auto i_edge(inertia_matrix["i_edge"].as<std::vector<double>>());
    const auto i_flap(inertia_matrix["i_flap"].as<std::vector<double>>());
    const auto i_plr(inertia_matrix["i_plr"].as<std::vector<double>>());
    const auto i_cp(inertia_matrix["i_cp"].as<std::vector<double>>());
    const auto n_sections = stiffness_matrix["grid"].size();

    if (m_grid.size() != k_grid.size()) {
        throw std::runtime_error("stiffness and mass matrices not on same grid");
    }
    for (auto i = 0U; i < n_sections; ++i) {
        if (abs(m_grid[i] - k_grid[i]) > 1e-8) {
            throw std::runtime_error("stiffness and mass matrices not on same grid");
        }
        builder.Blade().AddSection(
            m_grid[i],
            {{
                {mass[i], 0., 0., 0., 0., -mass[i] * cm_y[i]},
                {0., mass[i], 0., 0., 0., mass[i] * cm_x[i]},
                {0., 0., mass[i], mass[i] * cm_y[i], -mass[i] * cm_x[i], 0.},
                {0., 0., mass[i] * cm_y[i], i_edge[i], -i_cp[i], 0.},
                {0., 0., -mass[i] * cm_x[i], -i_cp[i], i_flap[i], 0.},
                {-mass[i] * cm_y[i], mass[i] * cm_x[i], 0., 0., 0., i_plr[i]},
            }},
            {{
                {k11[i], k12[i], k13[i], k14[i], k15[i], k16[i]},
                {k12[i], k22[i], k23[i], k24[i], k25[i], k26[i]},
                {k13[i], k23[i], k33[i], k34[i], k35[i], k36[i]},
                {k14[i], k24[i], k34[i], k44[i], k45[i], k46[i]},
                {k15[i], k25[i], k35[i], k45[i], k55[i], k56[i]},
                {k16[i], k26[i], k36[i], k46[i], k56[i], k66[i]},
            }},
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    // Build blade interface
    auto interface = builder.Build();

    // Write initial vtk output
    interface.WriteOutputVTK();

    // Get reference to tip node
    auto& tip_node = interface.Blade().nodes.back();

    // Loop through solution iterations
    for (auto i = 1U; i < 20U; ++i) {
        // Calculate time
        const auto t{static_cast<double>(i) * time_step};

        // Apply oscillating moment about z axis to tip
        tip_node.loads[1] = 2.0e5 * sin(2. * M_PI * 0.05 * t);

        // Take step
        const auto converged = interface.Step();

        // Check convergence
        ASSERT_EQ(converged, true);

        // Write vtk output
        interface.WriteOutputVTK();
    }

    EXPECT_NEAR(tip_node.position[0], 117.00012960730839, 1e-10);
    EXPECT_NEAR(tip_node.position[1], 0.1703315069675968, 1e-10);
    EXPECT_NEAR(tip_node.position[2], 4.0021291879233418, 1e-10);
    EXPECT_NEAR(tip_node.position[3], 0.9987613005270843, 1e-10);
    EXPECT_NEAR(tip_node.position[4], -0.001514812181144, 1e-10);
    EXPECT_NEAR(tip_node.position[5], -0.049439491482141, 1e-10);
    EXPECT_NEAR(tip_node.position[6], 0.0054135566396299, 1e-10);
}

TEST(BladeInterfaceTest, RotatingBeam) {
    const auto time_step{0.01};
    const Array_3 omega{0., 0., 1.};
    const Array_3 x0_root{2., 0., 0.};
    const auto root_vel = CrossProduct(omega, x0_root);

    // Create interface builder
    auto builder = interfaces::BladeInterfaceBuilder{};

    // Set solution options
    builder.Solution()
        .EnableDynamicSolve()
        .SetTimeStep(time_step)
        .SetDampingFactor(0.0)
        .SetMaximumNonlinearIterations(6)
        .SetAbsoluteErrorTolerance(1e-6)
        .SetRelativeErrorTolerance(1e-4);

#ifdef OpenTurbine_ENABLE_VTK
    builder.Solution().SetVTKOutputPath("BladeInterfaceTest.RotatingBeam/step_####");
#endif

    // Node locations (GLL quadrature)
    const auto node_s = std::vector{
        0., 0.11747233803526763, 0.35738424175967748, 0.64261575824032247, 0.88252766196473242, 1.
    };

    builder.Blade()
        .SetElementOrder(5)
        .PrescribedRootMotion(true)
        .SetRootPosition({x0_root[0], x0_root[1], x0_root[2], 1., 0., 0., 0.})
        .SetRootVelocity({root_vel[0], root_vel[1], root_vel[2], omega[0], omega[1], omega[2]})
        .AddRefAxisTwist(0., 0.)
        .AddRefAxisTwist(1., 0.);

    // Add reference axis coordinates and twist
    for (const auto s : node_s) {
        builder.Blade().AddRefAxisPoint(
            s, {10. * s, 0., 0.}, interfaces::components::ReferenceAxisOrientation::X
        );
    }

    // Beam section locations
    const std::vector<double> section_s{
        0.0254460438286208, 0.1292344072003028, 0.2970774243113015, 0.5,
        0.7029225756886985, 0.8707655927996972, 0.9745539561713792,
    };

    // Add reference axis coordinates and twist
    for (const auto s : section_s) {
        builder.Blade().AddSection(
            s,
            std::array{
                std::array{8.538e-2, 0., 0., 0., 0., 0.},
                std::array{0., 8.538e-2, 0., 0., 0., 0.},
                std::array{0., 0., 8.538e-2, 0., 0., 0.},
                std::array{0., 0., 0., 1.4433e-2, 0., 0.},
                std::array{0., 0., 0., 0., 0.40972e-2, 0.},
                std::array{0., 0., 0., 0., 0., 1.0336e-2},
            },
            std::array{
                std::array{1368.17e3, 0., 0., 0., 0., 0.},
                std::array{0., 88.56e3, 0., 0., 0., 0.},
                std::array{0., 0., 38.78e3, 0., 0., 0.},
                std::array{0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
                std::array{0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
                std::array{0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
            },
            interfaces::components::ReferenceAxisOrientation::X
        );
    }

    // Create the interface
    auto interface = builder.Build();

    // Loop through time steps
    for (auto i = 1U; i < 10U; ++i) {
        // Calculate time at end of step
        const auto t{static_cast<double>(i) * time_step};

        // Calculate root displacement from initial position and apply
        const auto u_rot = RotationVectorToQuaternion({omega[0] * t, omega[1] * t, omega[2] * t});
        const auto x_root = RotateVectorByQuaternion(u_rot, x0_root);
        const Array_3 u_trans{
            x_root[0] - x0_root[0], x_root[1] - x0_root[1], x_root[2] - x0_root[2]
        };
        interface.SetRootDisplacement(
            {u_trans[0], u_trans[1], u_trans[2], u_rot[0], u_rot[1], u_rot[2], u_rot[3]}
        );

        // Calculate state at end of step
        const auto converged = interface.Step();

        // Check for convergence
        ASSERT_EQ(converged, true);

        // Write VTK file for end of step
        interface.WriteOutputVTK();
    }

    //--------------------------------------------------------------------------
    // Root Node
    //--------------------------------------------------------------------------

    EXPECT_NEAR(interface.Blade().root_node.position[0], 1.9919054660239885, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.position[1], 0.17975709839602211, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.position[2], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.position[3], 0.99898767084784234, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.position[4], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.position[5], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.position[6], 0.044984814037660227, 1e-10);

    EXPECT_NEAR(interface.Blade().root_node.displacement[0], -0.0080945339760114532, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.displacement[1], 0.17975709839602211, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.displacement[2], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.displacement[3], 0.99898767084784234, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.displacement[4], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.displacement[5], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.displacement[6], 0.044984814037660227, 1e-10);

    EXPECT_NEAR(interface.Blade().root_node.velocity[0], -0.17976259212149071, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.velocity[1], 1.9919719054875213, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.velocity[2], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.velocity[3], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.velocity[4], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.velocity[5], 1., 1e-10);

    EXPECT_NEAR(interface.Blade().root_node.acceleration[0], -1.9920882238788629, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.acceleration[1], -0.17977158309645525, 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.acceleration[2], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.acceleration[3], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.acceleration[4], 0., 1e-10);
    EXPECT_NEAR(interface.Blade().root_node.acceleration[5], 0., 1e-10);

    //--------------------------------------------------------------------------
    // Tip Node
    //--------------------------------------------------------------------------

    EXPECT_NEAR(interface.Blade().nodes[5].position[0], 11.951460424343411, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].position[1], 1.0785463019560009, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].position[2], 2.2845605555050727E-10, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].position[3], 0.99898766738195299, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].position[4], 2.6570959740959799E-9, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].position[5], -1.0123941616837804E-10, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].position[6], 0.044984891005365443, 1e-10);

    EXPECT_NEAR(interface.Blade().nodes[5].displacement[0], -0.048539575656588806, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].displacement[1], 1.0785463019560009, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].displacement[2], 2.2845605555050727E-10, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].displacement[3], 0.99898766738195299, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].displacement[4], 2.6570959740959799E-9, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].displacement[5], -1.0123941616837804E-10, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].displacement[6], 0.044984891005365443, 1e-10);

    EXPECT_NEAR(interface.Blade().nodes[5].velocity[0], -1.0786835619583102, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].velocity[1], 11.95303967174639, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].velocity[2], -6.3862048021175104E-8, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].velocity[3], 0.000003159329892760129, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].velocity[4], 2.5866107917084571E-7, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].velocity[5], 1.0001562024381443, 1e-10);

    EXPECT_NEAR(interface.Blade().nodes[5].acceleration[0], -11.955952417551353, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].acceleration[1], -1.0800839943831992, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].acceleration[2], -0.000013544534439173783, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].acceleration[3], -0.000046709402446003812, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].acceleration[4], 0.000008256203749646939, 1e-10);
    EXPECT_NEAR(interface.Blade().nodes[5].acceleration[5], 0.00020635762959181575, 1e-10);
}

TEST(BladeInterfaceTest, StaticCurledBeam) {
    // Create interface builder
    auto builder = interfaces::BladeInterfaceBuilder{};

    builder.Solution()
        .EnableStaticSolve()
        .SetTimeStep(1.)
        .SetDampingFactor(1.)
        .SetMaximumNonlinearIterations(10)
        .SetAbsoluteErrorTolerance(1e-5)
        .SetRelativeErrorTolerance(1e-3);

#ifdef OpenTurbine_ENABLE_VTK
    builder.Solution().SetVTKOutputPath("BladeInterfaceTest.StaticCurledBeam/step_####");
#endif

    // Node locations
    const std::vector<double> kp_s{0., 1.};

    builder.Blade()
        .SetElementOrder(10)
        .SetSectionRefinement(1)
        .PrescribedRootMotion(true)
        .AddRefAxisTwist(0., 0.)
        .AddRefAxisTwist(1., 0.);

    for (const auto s : kp_s) {
        builder.Blade().AddRefAxisPoint(
            s, {s * 10., 0., 0.}, interfaces::components::ReferenceAxisOrientation::X
        );
    }

    // Beam section locations
    const std::vector<double> section_s{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.};

    // Add reference axis coordinates and twist
    for (const auto s : section_s) {
        builder.Blade().AddSection(
            s,
            std::array{
                std::array{1., 0., 0., 0., 0., 0.},
                std::array{0., 1., 0., 0., 0., 0.},
                std::array{0., 0., 1., 0., 0., 0.},
                std::array{0., 0., 0., 1., 0., 0.},
                std::array{0., 0., 0., 0., 1., 0.},
                std::array{0., 0., 0., 0., 0., 1.},
            },
            std::array{
                std::array{1770.e3, 0., 0., 0., 0., 0.},
                std::array{0., 1770.e3, 0., 0., 0., 0.},
                std::array{0., 0., 1770.e3, 0., 0., 0.},
                std::array{0., 0., 0., 8.16e3, 0., 0.},
                std::array{0., 0., 0., 0., 86.9e3, 0.},
                std::array{0., 0., 0., 0., 0., 215.e3},
            },
            interfaces::components::ReferenceAxisOrientation::X
        );
    }

    auto interface = builder.Build();

    // Create vector to store deformed tip positions
    std::vector<Array_3> tip_positions;

    // Get reference to tip node
    auto& tip_node = interface.Blade().nodes[interface.Blade().nodes.size() - 1];

    // Loop through moments to apply to tip
    const std::vector<double> moments{0., 10920.0, 21840.0, 32761.0, 43681.0, 54601.0};
    for (const auto m : moments) {
        // Apply moment to tip about y axis
        tip_node.loads[4] = -m;

        // Static step
        const auto converged = interface.Step();

        // Write beam visualization output
        interface.WriteOutputVTK();

        // Check convergence
        ASSERT_EQ(converged, true);

        // Add tip position
        tip_positions.emplace_back(
            Array_3{tip_node.position[0], tip_node.position[1], tip_node.position[2]}
        );
    }

    EXPECT_NEAR(tip_positions[0][0], 10., 1e-8);
    EXPECT_NEAR(tip_positions[0][1], 0., 1e-8);
    EXPECT_NEAR(tip_positions[0][2], 0., 1e-8);

    EXPECT_NEAR(tip_positions[1][0], 7.535457547469286, 1e-8);
    EXPECT_NEAR(tip_positions[1][1], 0., 1e-8);
    EXPECT_NEAR(tip_positions[1][2], 5.5405833775092788, 1e-8);

    EXPECT_NEAR(tip_positions[2][0], 2.275140291113245, 1e-8);
    EXPECT_NEAR(tip_positions[2][1], 0., 1e-8);
    EXPECT_NEAR(tip_positions[2][2], 7.2175190489085246, 1e-8);

    EXPECT_NEAR(tip_positions[3][0], -1.6157938054255538, 1e-8);
    EXPECT_NEAR(tip_positions[3][1], 0., 1e-8);
    EXPECT_NEAR(tip_positions[3][2], 4.7783647698451075, 1e-8);

    EXPECT_NEAR(tip_positions[4][0], -1.9061447828587319, 1e-8);
    EXPECT_NEAR(tip_positions[4][1], 0., 1e-8);
    EXPECT_NEAR(tip_positions[4][2], 1.332200967141842, 1e-8);

    EXPECT_NEAR(tip_positions[5][0], 0.022656037313893762, 1e-8);
    EXPECT_NEAR(tip_positions[5][1], 0., 1e-8);
    EXPECT_NEAR(tip_positions[5][2], 0.0022466646330885354, 1e-8);
}

}  // namespace openturbine::tests
