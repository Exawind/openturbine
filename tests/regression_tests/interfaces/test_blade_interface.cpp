#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "interfaces/blade/blade_interface_builder.hpp"
#include "regression/test_utilities.hpp"

namespace openturbine::tests {

using namespace openturbine::interfaces;
using namespace openturbine::interfaces::components;

BladeInterfaceBuilder BuilderFromWindIO() {
    const YAML::Node windio = YAML::LoadFile("interfaces_test_files/IEA-15-240-RWT.yaml");

    // Create interface builder
    auto builder = BladeInterfaceBuilder{};

    // Add reference axis coordinates and twist
    const auto blade_axis = windio["components"]["blade"]["reference_axis"];
    const auto ref_axis_n_coord_points = blade_axis["grid"].size();
    for (auto i = 0U; i < ref_axis_n_coord_points; ++i) {
        builder.Blade().AddRefAxisPointZ(
            blade_axis["grid"][i].as<double>(),
            {
                blade_axis["x"][i].as<double>(),
                blade_axis["y"][i].as<double>(),
                blade_axis["z"][i].as<double>(),
            }
        );
    }
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
        builder.Blade().AddSectionRefZ(
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
            }}
        );
    }

    return builder;
}

TEST(BladeInterfaceTest, TestBuilder) {
    auto builder = BuilderFromWindIO();
    builder.Blade().PrescribedRootMotion(true);

    auto interface = builder.Build();

    for (auto i = 0U; i < 5; ++i) {
        auto converged = interface.Step();
        ASSERT_EQ(converged, true);
    }
}

TEST(BladeInterfaceTest, RotatingBlade1) {
    const Array_3 omega{0., 1., 0.};
    const auto time_step{0.01};

    auto builder = BuilderFromWindIO();
    builder.Blade().PrescribedRootMotion(true).SetRootVelocity(
        {0., 0., 0., omega[0], omega[1], omega[2]}
    );

    builder.Solution()
        .SetTimeStep(time_step)
        .EnableDynamicSolve()
        .SetDampingFactor(0.0)
        .SetMaximumNonlinearIterations(5);

    auto interface = builder.Build();

    for (auto i = 1U; i < 5U; ++i) {
        auto u_rot = RotationVectorToQuaternion(
            {omega[0] * i * time_step, omega[1] * i * time_step, omega[2] * i * time_step}
        );
        interface.SetRootDisplacement({0., 0., 0., u_rot[0], u_rot[1], u_rot[2], u_rot[3]});
        auto converged = interface.Step();
        ASSERT_EQ(converged, true);
    }

    ASSERT_DOUBLE_EQ(interface.blade.root_node.position[0], 0.);
    ASSERT_DOUBLE_EQ(interface.blade.root_node.position[1], 0.);
    ASSERT_DOUBLE_EQ(interface.blade.root_node.position[2], 0.);
    ASSERT_DOUBLE_EQ(interface.blade.root_node.position[3], 0.99980000666657776);
    ASSERT_DOUBLE_EQ(interface.blade.root_node.position[4], 0.);
    ASSERT_DOUBLE_EQ(interface.blade.root_node.position[5], 0.01999866669333308);
    ASSERT_DOUBLE_EQ(interface.blade.root_node.position[6], 0.);
}

TEST(BladeInterfaceTest, RotatingBlade2) {
    const auto time_step{0.01};
    const Array_3 omega{0., 0., 1.};
    const Array_3 x0_root{0., 0., 0.};
    const auto root_vel = CrossProduct(omega, x0_root);

    auto builder = BuilderFromWindIO();

    builder.Blade()
        .PrescribedRootMotion(true)
        .SetRootPosition({x0_root[0], x0_root[1], x0_root[2], 1., 0., 0., 0.})
        .SetRootVelocity({root_vel[0], root_vel[1], root_vel[2], omega[0], omega[1], omega[2]});

    builder.Solution()
        .SetTimeStep(time_step)
        .EnableDynamicSolve()
        .SetDampingFactor(0.0)
        .SetMaximumNonlinearIterations(6)
        .SetAbsoluteErrorTolerance(1e-6)
        .SetRelativeErrorTolerance(1e-4)
        .SetVTKOutputPath("BladeInterfaceTest.RotatingBlade2/step_####.vtu");

    auto interface = builder.Build();

    for (auto i = 1U; i < 1000U; ++i) {
        const auto t{static_cast<double>(i) * time_step};
        auto u_rot = RotationVectorToQuaternion({omega[0] * t, omega[1] * t, omega[2] * t});
        auto x_root = RotateVectorByQuaternion(u_rot, x0_root);
        Array_3 u_trans{x_root[0] - x0_root[0], x_root[1] - x0_root[1], x_root[2] - x0_root[2]};
        interface.SetRootDisplacement(
            {u_trans[0], u_trans[1], u_trans[2], u_rot[0], u_rot[1], u_rot[2], u_rot[3]}
        );
        auto converged = interface.Step();
        ASSERT_EQ(converged, true);
        interface.WriteOutputVTK();
    }

    // ASSERT_DOUBLE_EQ(interface.blade.root_node.position[0], 1.9800166611121033);
    // ASSERT_DOUBLE_EQ(interface.blade.root_node.position[1], 0.39933366658731262);
    // ASSERT_DOUBLE_EQ(interface.blade.root_node.position[2], 0.);
    // ASSERT_DOUBLE_EQ(interface.blade.root_node.position[3], 0.99875026039496628);
    // ASSERT_DOUBLE_EQ(interface.blade.root_node.position[4], 0.);
    // ASSERT_DOUBLE_EQ(interface.blade.root_node.position[5], 0.);
    // ASSERT_DOUBLE_EQ(interface.blade.root_node.position[6], 0.049979169270678331);
}

}  // namespace openturbine::tests
