#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "interfaces/components/inflow.hpp"

namespace openturbine::tests {

struct TestCase {
    double time;
    std::array<double, 3> position;
    std::array<double, 3> vel_exp;
};

TEST(InflowTest, SteadyWindWithoutShear) {
    // Define steady 10 m/s wind along X-axis for any time and position
    constexpr auto vel_h = 10.;
    constexpr auto ref_height = 100.;
    constexpr auto power_law_exp = 0.;
    constexpr auto flow_angle_horizontal = 0.;
    auto inflow = interfaces::components::Inflow::SteadyWind(
        vel_h, ref_height, power_law_exp, flow_angle_horizontal
    );

    constexpr auto test_cases = std::array{
        TestCase{0., {0., 0., 0.}, {10., 0., 0.}},  // Test case at time 0 and at ground level
        TestCase{
            1., {0., 0., ref_height}, {10., 0., 0.}
        },  // Test case at time 1 and at reference height
        TestCase{1000., {100., 100., 100.}, {10., 0., 0.}}  // Test case at time 1000 and at far away
    };

    for (const auto& test_case : test_cases) {
        auto velocity = inflow.Velocity(test_case.time, test_case.position);
        EXPECT_NEAR(velocity[0], test_case.vel_exp[0], 1e-12);
        EXPECT_NEAR(velocity[1], test_case.vel_exp[1], 1e-12);
        EXPECT_NEAR(velocity[2], test_case.vel_exp[2], 1e-12);
    }
}

TEST(InflowTest, SteadyWindWithShearNonzeroFlowAngle) {
    // Define steady 10 m/s wind along X-axis at ref height with 0.1 power law shear exponent
    constexpr auto vel_h = 10.;
    constexpr auto ref_height = 100.;
    constexpr auto power_law_exp = 0.1;
    constexpr auto flow_angle_horizontal = M_PI / 4.;  // 45 degrees -> radians
    auto inflow = openturbine::interfaces::components::Inflow::SteadyWind(
        vel_h, ref_height, power_law_exp, flow_angle_horizontal
    );

    constexpr auto test_cases = std::array{
        // Test case at time 0 and at reference height
        TestCase{0., {0., 0., ref_height}, {7.0710678118654755, -7.0710678118654755, 0.}},
        // Test case at time 1 and at ground level
        TestCase{1., {0., 0., 0.}, {0., 0., 0.}},
        // Test case at time 100 and at half ref height
        TestCase{100., {100., 100., ref_height / 2.}, {6.597539553864471, -6.597539553864471, 0.}}
    };

    for (const auto& test_case : test_cases) {
        const auto velocity = inflow.Velocity(test_case.time, test_case.position);
        EXPECT_NEAR(velocity[0], test_case.vel_exp[0], 1e-12);
        EXPECT_NEAR(velocity[1], test_case.vel_exp[1], 1e-12);
        EXPECT_NEAR(velocity[2], test_case.vel_exp[2], 1e-12);
    }
}

TEST(InflowTest, SteadyWindWithShear) {
    // Define steady 10 m/s wind along X-axis at ref height with 0.1 power law shear exponent
    constexpr auto vel_h = 10.;
    constexpr auto ref_height = 100.;
    constexpr auto power_law_exp = 0.1;
    constexpr auto flow_angle_horizontal = 0.;
    auto inflow = openturbine::interfaces::components::Inflow::SteadyWind(
        vel_h, ref_height, power_law_exp, flow_angle_horizontal
    );

    constexpr auto test_cases =
        std::array{// Test case at time 0 and at reference height
                   TestCase{0., {0., 0., ref_height}, {10., 0., 0.}},
                   // Test case at time 1 and at ground level
                   TestCase{1., {0., 0., 0.}, {0., 0., 0.}},
                   // Test case at time 100 and at half ref height
                   TestCase{100., {100., 100., ref_height / 2.}, {9.330329915368074, 0., 0.}}
        };

    for (const auto& test_case : test_cases) {
        const auto velocity = inflow.Velocity(test_case.time, test_case.position);
        EXPECT_NEAR(velocity[0], test_case.vel_exp[0], 1e-12);
        EXPECT_NEAR(velocity[1], test_case.vel_exp[1], 1e-12);
        EXPECT_NEAR(velocity[2], test_case.vel_exp[2], 1e-12);
    }
}

}  // namespace openturbine::tests
