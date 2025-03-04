#include <gtest/gtest.h>

#include "interfaces/cfd/interface.hpp"
#include "regression/test_utilities.hpp"
#include "viz/vtk_lines.hpp"

namespace openturbine::tests {

using namespace openturbine::cfd;

TEST(CFDInterfaceTest, PrecessionTest) {
    // Create cfd interface
    Interface interface(InterfaceInput{
        {{0., 0., 0.}},  // gravity
        0.01,            // time step
        1.,              // rho infinity (numerical damping)
        5,               // max convergence iterations
        TurbineInput{
            FloatingPlatformInput{
                true,                          // enable
                {0., 0., 0., 1., 0., 0., 0.},  // position
                {0., 0., 0., 0.5, 0.5, 1.0},   // velocity
                {0., 0., 0., 0., 0., 0.},      // acceleration
                {{
                    {1., 0., 0., 0., 0., 0.},   //
                    {0., 1., 0., 0., 0., 0.},   //
                    {0., 0., 1., 0., 0., 0.},   //
                    {0., 0., 0., 1., 0., 0.},   //
                    {0., 0., 0., 0., 1., 0.},   //
                    {0., 0., 0., 0., 0., 0.5},  //
                }},                             // platform mass matrix
                {},                             // no mooring lines
            },
        },
    });

    // Create reference to platform node in interface
    auto& platform_node = interface.turbine.floating_platform.node;

    // Run simulation for 500 steps
    for (size_t i = 0; i < 500; ++i) {
        EXPECT_EQ(interface.Step(), true);
    }

    // Check results at 500 steps
    EXPECT_NEAR(platform_node.displacement[0], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[1], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[2], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[3], -0.6305304765029902, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[4], 0.6055602536398981, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[5], -0.30157705376951366, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[6], -0.3804988542061519, 1.e-12);

    // Save the current state
    interface.SaveState();

    // Run simulation for an additional 100 steps
    for (size_t i = 500; i < 600; ++i) {
        EXPECT_EQ(interface.Step(), true);
    }

    // Check results at 600 steps
    EXPECT_NEAR(platform_node.displacement[0], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[1], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[2], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[3], -0.35839726967749647, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[4], 0.31963473392384162, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[5], -0.2758730482813182, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[6], -0.83263383019736148, 1.e-12);

    // Restore to saved state at 500 steps
    interface.RestoreState();

    // Check that results match 500 steps
    EXPECT_NEAR(platform_node.displacement[0], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[1], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[2], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[3], -0.6305304765029902, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[4], 0.6055602536398981, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[5], -0.30157705376951366, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[6], -0.3804988542061519, 1.e-12);

    // Run simulation from 500 to 600 steps
    for (size_t i = 500; i < 600; ++i) {
        EXPECT_EQ(interface.Step(), true);
    }

    // Check that simulation gives same results at 600 steps
    EXPECT_NEAR(platform_node.displacement[0], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[1], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[2], 0., 1.e-12);
    EXPECT_NEAR(platform_node.displacement[3], -0.35839726967749647, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[4], 0.31963473392384162, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[5], -0.2758730482813182, 1.e-12);
    EXPECT_NEAR(platform_node.displacement[6], -0.83263383019736148, 1.e-12);
}

#ifdef OpenTurbine_ENABLE_VTK
void OutputLines(const FloatingPlatform& platform, size_t step_num, const std::string& output_dir) {
    auto tmp = std::to_string(step_num);
    auto step_num_str = std::string(5 - tmp.size(), '0') + tmp;
    WriteLinesVTK(
        {
            {0, 1},
            {0, 2},
            {0, 3},
        },
        {
            platform.node.position,
            platform.mooring_lines[0].fairlead_node.position,
            platform.mooring_lines[1].fairlead_node.position,
            platform.mooring_lines[2].fairlead_node.position,
        },
        output_dir + "/platform_" + step_num_str
    );

    WriteLinesVTK(
        {
            {0, 1},
            {2, 3},
            {4, 5},
        },
        {
            platform.mooring_lines[0].fairlead_node.position,
            platform.mooring_lines[0].anchor_node.position,
            platform.mooring_lines[1].fairlead_node.position,
            platform.mooring_lines[1].anchor_node.position,
            platform.mooring_lines[2].fairlead_node.position,
            platform.mooring_lines[2].anchor_node.position,
        },
        output_dir + "/mooring_" + step_num_str
    );
}
#else
void OutputLines(const FloatingPlatform&, size_t, const std::string&) {
}
#endif

TEST(CFDInterfaceTest, FloatingPlatform) {
    // Solution parameters
    constexpr auto time_step = 0.01;
    constexpr auto t_end = 1.;
    constexpr auto rho_inf = 0.0;
    constexpr auto max_iter = 5;
    const auto n_steps{static_cast<size_t>(ceil(t_end / time_step)) + 1};

    // Construct platform mass matrix
    constexpr auto platform_mass{1.419625E+7};                           // kg
    constexpr Array_3 gravity{0., 0., -9.8124};                          // m/s/s
    constexpr Array_3 platform_moi{1.2898E+10, 1.2851E+10, 1.4189E+10};  // kg*m*m
    constexpr Array_3 platform_cm_position{0., 0., -7.53};               // m
    constexpr Array_6x6 platform_mass_matrix{{
        {platform_mass, 0., 0., 0., 0., 0.},    // Row 1
        {0., platform_mass, 0., 0., 0., 0.},    // Row 2
        {0., 0., platform_mass, 0., 0., 0.},    // Row 3
        {0., 0., 0., platform_moi[0], 0., 0.},  // Row 4
        {0., 0., 0., 0., platform_moi[1], 0.},  // Row 5
        {0., 0., 0., 0., 0., platform_moi[2]},  // Row 6
    }};

    // Mooring line properties
    constexpr auto mooring_line_stiffness{48.9e3};       // N
    constexpr auto mooring_line_initial_length{55.432};  // m

    // Create cfd interface
    Interface interface(InterfaceInput{
        gravity,
        time_step,  // time step
        rho_inf,    // rho infinity (numerical damping)
        max_iter,   // max convergence iterations
        TurbineInput{
            FloatingPlatformInput{
                true,  // enable
                {
                    platform_cm_position[0],
                    platform_cm_position[1],
                    platform_cm_position[2],
                    1.,
                    0.,
                    0.,
                    0.,
                },                         // position
                {0., 0., 0., 0., 0., 0.},  // velocity
                {0., 0., 0., 0., 0., 0.},  // acceleration
                platform_mass_matrix,
                {
                    {
                        mooring_line_stiffness,
                        mooring_line_initial_length,
                        {-40.87, 0.0, -14.},    // Fairlead node coordinates
                        {0., 0., 0.},           // Fairlead node velocity
                        {0., 0., 0.},           // Fairlead node acceleration
                        {-105.47, 0.0, -58.4},  // Anchor node coordinates
                        {0., 0., 0.},           // Anchor node velocity
                        {0., 0., 0.},           // Anchor node acceleration
                    },
                    {
                        mooring_line_stiffness,
                        mooring_line_initial_length,
                        {20.43, -35.39, -14.},   // Fairlead node coordinates
                        {0., 0., 0.},            // Fairlead node velocity
                        {0., 0., 0.},            // Fairlead node acceleration
                        {52.73, -91.34, -58.4},  // Anchor node coordinates
                        {0., 0., 0.},            // Anchor node velocity
                        {0., 0., 0.},            // Anchor node acceleration
                    },
                    {
                        mooring_line_stiffness,
                        mooring_line_initial_length,
                        {20.43, 35.39, -14.},   // Fairlead node coordinates
                        {0., 0., 0.},           // Fairlead node velocity
                        {0., 0., 0.},           // Fairlead node acceleration
                        {52.73, 91.34, -58.4},  // Anchor node coordinates
                        {0., 0., 0.},           // Anchor node velocity
                        {0., 0., 0.},           // Anchor node acceleration
                    },
                },
            },
        },
    });

    // Calculate buoyancy force as percentage of gravitational force plus spring forces times
    const auto initial_spring_force = 1907514.4912628897;
    const auto platform_gravity_force = -gravity[2] * platform_mass;
    const auto buoyancy_force = initial_spring_force + platform_gravity_force;

    const std::string output_dir{"FloatingPlatform"};
    RemoveDirectoryWithRetries(output_dir);
    std::filesystem::create_directory(output_dir);

    // Iterate through time steps
    for (size_t i = 0U; i < n_steps; ++i) {
        // Calculate current time
        const auto t = static_cast<double>(i) * time_step;

        // Write VTK visualization output
        OutputLines(interface.turbine.floating_platform, i, output_dir);

        // Apply load in y direction
        interface.turbine.floating_platform.node.loads[1] = 1e6 * sin(2. * M_PI / 20. * t);

        // Apply time varying buoyancy force
        interface.turbine.floating_platform.node.loads[2] =
            buoyancy_force + 0.5 * initial_spring_force * sin(2. * M_PI / 20. * t);

        // Apply time varying moments to platform node
        interface.turbine.floating_platform.node.loads[3] = 5.0e5 * sin(2. * M_PI / 15. * t);  // rx
        interface.turbine.floating_platform.node.loads[4] = 1.0e6 * sin(2. * M_PI / 30. * t);  // ry
        interface.turbine.floating_platform.node.loads[5] = 2.0e7 * sin(2. * M_PI / 60. * t);  // rz

        // Step
        const auto converged = interface.Step();
        EXPECT_EQ(converged, true);

        // Check for expected displacements/rotations of platform node
        if (i == 100) {
            const auto& platform_u = interface.turbine.floating_platform.node.displacement;
            EXPECT_NEAR(platform_u[0], 3.4379940641493395e-06, 1e-12);
            EXPECT_NEAR(platform_u[1], 0.0036709089002912518, 1e-12);
            EXPECT_NEAR(platform_u[2], 0.0034998271929752248, 1e-12);
            EXPECT_NEAR(platform_u[3], 0.99999999992274302, 1e-12);
            EXPECT_NEAR(platform_u[4], 1.3448158662549561e-06, 1e-12);
            EXPECT_NEAR(platform_u[5], 1.2713890988278111e-06, 1e-12);
            EXPECT_NEAR(platform_u[6], 1.2291853012439073e-05, 1e-12);
        }
    }
}

TEST(CFDInterfaceTest, Restart) {
    // Solution parameters
    constexpr auto time_step = 0.01;
    constexpr auto rho_inf = 0.0;
    constexpr auto max_iter = 5;

    // Construct platform mass matrix
    constexpr auto platform_mass{1.419625E+7};                           // kg
    constexpr Array_3 gravity{0., 0., -9.8124};                          // m/s/s
    constexpr Array_3 platform_moi{1.2898E+10, 1.2851E+10, 1.4189E+10};  // kg*m*m
    constexpr Array_3 platform_cm_position{0., 0., -7.53};               // m
    constexpr Array_6x6 platform_mass_matrix{{
        {platform_mass, 0., 0., 0., 0., 0.},    // Row 1
        {0., platform_mass, 0., 0., 0., 0.},    // Row 2
        {0., 0., platform_mass, 0., 0., 0.},    // Row 3
        {0., 0., 0., platform_moi[0], 0., 0.},  // Row 4
        {0., 0., 0., 0., platform_moi[1], 0.},  // Row 5
        {0., 0., 0., 0., 0., platform_moi[2]},  // Row 6
    }};

    // Mooring line properties
    constexpr auto mooring_line_stiffness{48.9e3};       // N
    constexpr auto mooring_line_initial_length{55.432};  // m

    auto interface_input = InterfaceInput{
        gravity,
        time_step,  // time step
        rho_inf,    // rho infinity (numerical damping)
        max_iter,   // max convergence iterations
        TurbineInput{
            FloatingPlatformInput{
                true,  // enable
                {
                    platform_cm_position[0],
                    platform_cm_position[1],
                    platform_cm_position[2],
                    1.,
                    0.,
                    0.,
                    0.,
                },                         // position
                {0., 0., 0., 0., 0., 0.},  // velocity
                {0., 0., 0., 0., 0., 0.},  // acceleration
                platform_mass_matrix,
                {
                    {
                        mooring_line_stiffness,
                        mooring_line_initial_length,
                        {-40.87, 0.0, -14.},    // Fairlead node coordinates
                        {0., 0., 0.},           // Fairlead node velocity
                        {0., 0., 0.},           // Fairlead node acceleration
                        {-105.47, 0.0, -58.4},  // Anchor node coordinates
                        {0., 0., 0.},           // Anchor node velocity
                        {0., 0., 0.},           // Anchor node acceleration
                    },
                    {
                        mooring_line_stiffness,
                        mooring_line_initial_length,
                        {20.43, -35.39, -14.},   // Fairlead node coordinates
                        {0., 0., 0.},            // Fairlead node velocity
                        {0., 0., 0.},            // Fairlead node acceleration
                        {52.73, -91.34, -58.4},  // Anchor node coordinates
                        {0., 0., 0.},            // Anchor node velocity
                        {0., 0., 0.},            // Anchor node acceleration
                    },
                    {
                        mooring_line_stiffness,
                        mooring_line_initial_length,
                        {20.43, 35.39, -14.},   // Fairlead node coordinates
                        {0., 0., 0.},           // Fairlead node velocity
                        {0., 0., 0.},           // Fairlead node acceleration
                        {52.73, 91.34, -58.4},  // Anchor node coordinates
                        {0., 0., 0.},           // Anchor node velocity
                        {0., 0., 0.},           // Anchor node acceleration
                    },
                },
            },
        },
    };

    auto interface1 = Interface(interface_input);

    // Take 10 initial steps
    for (auto i = 0U; i < 100U; ++i) {
        const auto t = static_cast<double>(i) * time_step;
        interface1.turbine.floating_platform.node.loads[1] = 1e6 * sin(2. * M_PI / 20. * t);
        auto converged = interface1.Step();
        EXPECT_TRUE(converged);
    }

    interface1.WriteRestart("test_restart.dat");

    // Take 10 more steps using original system
    for (auto i = 0U; i < 100U; ++i) {
        const auto t = static_cast<double>(i) * time_step;
        interface1.turbine.floating_platform.node.loads[1] = 1e6 * sin(2. * M_PI / 20. * t);
        auto converged = interface1.Step();
        EXPECT_TRUE(converged);
    }

    auto interface2 = Interface(interface_input);
    interface2.ReadRestart("test_restart.dat");

    // Take 10 steps using restarted system
    for (auto i = 0U; i < 100U; ++i) {
        const auto t = static_cast<double>(i) * time_step;
        interface2.turbine.floating_platform.node.loads[1] = 1e6 * sin(2. * M_PI / 20. * t);
        auto converged = interface2.Step();
        EXPECT_TRUE(converged);
    }

    const auto& platform_u_1 = interface1.turbine.floating_platform.node.displacement;
    const auto& platform_u_2 = interface2.turbine.floating_platform.node.displacement;
    // Ensure platform location is the same for original and restarted system
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(platform_u_1[i], platform_u_2[i]);
    }

    std::filesystem::remove("test_restart.dat");
}
}  // namespace openturbine::tests
