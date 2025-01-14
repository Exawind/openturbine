#include <gtest/gtest.h>

#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/interfaces/cfd/interface_ot.hpp"
#include "src/model/model.hpp"
#include "src/state/set_node_external_loads.hpp"
#include "src/state/state.hpp"
#include "src/step/step.hpp"
#include "src/step/update_system_variables.hpp"
#include "src/types.hpp"

namespace openturbine::tests {

using namespace openturbine::cfd;

TEST(CFDInterfaceTest, PrecessionTest) {
    // Create cfd interface
    InterfaceOT interface(InterfaceInput{
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
        interface.Step();
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
        interface.Step();
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
        interface.Step();
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

}  // namespace openturbine::tests
