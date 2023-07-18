#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(MathUtilitiesTest, CloseTo) {
    ASSERT_TRUE(close_to(1., 1.));
    ASSERT_TRUE(close_to(1., 1. + 1e-7));
    ASSERT_TRUE(close_to(1., 1. - 1e-7));
    ASSERT_FALSE(close_to(1., 1. + 1e-5));
    ASSERT_FALSE(close_to(1., 1. - 1e-5));
    ASSERT_TRUE(close_to(1e-7, 1e-7));

    ASSERT_TRUE(close_to(-1., -1.));
    ASSERT_TRUE(close_to(-1., -1. + 1e-7));
    ASSERT_TRUE(close_to(-1., -1. - 1e-7));
    ASSERT_FALSE(close_to(-1., -1. + 1e-5));
    ASSERT_FALSE(close_to(-1., -1. - 1e-5));
    ASSERT_TRUE(close_to(-1e-7, -1e-7));

    ASSERT_FALSE(close_to(1., -1.));
    ASSERT_FALSE(close_to(-1., 1.));
    ASSERT_FALSE(close_to(1., -1. + 1e-7));
    ASSERT_FALSE(close_to(-1., 1. + 1e-7));
    ASSERT_FALSE(close_to(1., -1. - 1e-7));
    ASSERT_FALSE(close_to(-1., 1. - 1e-7));
    ASSERT_FALSE(close_to(1., -1. + 1e-5));
    ASSERT_FALSE(close_to(-1., 1. + 1e-5));
    ASSERT_FALSE(close_to(1., -1. - 1e-5));
    ASSERT_FALSE(close_to(-1., 1. - 1e-5));
}

TEST(MathUtilitiesTest, WrapAngleToPi) {
    // 0 degrees
    ASSERT_NEAR(wrap_angle_to_pi(0), 0., 1e-6);
    // 45 degrees
    ASSERT_NEAR(wrap_angle_to_pi(kPI / 4.), kPI / 4., 1e-6);
    // -45 degrees
    ASSERT_NEAR(wrap_angle_to_pi(-kPI / 4.), -kPI / 4., 1e-6);
    // 90 degrees
    ASSERT_NEAR(wrap_angle_to_pi(0.5 * kPI), 0.5 * kPI, 1e-6);
    // -90 degrees
    ASSERT_NEAR(wrap_angle_to_pi(-0.5 * kPI), -0.5 * kPI, 1e-6);
    // 135 degrees
    ASSERT_NEAR(wrap_angle_to_pi(kPI / 2. + kPI / 4.), 0.75 * kPI, 1e-6);
    // -135 degrees
    ASSERT_NEAR(wrap_angle_to_pi(-kPI / 2. - kPI / 4.), -0.75 * kPI, 1e-6);
    // 180 degrees
    ASSERT_NEAR(wrap_angle_to_pi(kPI), kPI, 1e-6);
    // -180 degrees
    ASSERT_NEAR(wrap_angle_to_pi(-kPI), -kPI, 1e-6);
    // 225 degrees
    ASSERT_NEAR(wrap_angle_to_pi(kPI + kPI / 4.), -0.75 * kPI, 1e-6);
    // -225 degrees
    ASSERT_NEAR(wrap_angle_to_pi(-kPI - kPI / 4.), 0.75 * kPI, 1e-6);
    // 270 degrees
    ASSERT_NEAR(wrap_angle_to_pi(3. * kPI / 2.), -0.5 * kPI, 1e-6);
    // -270 degrees
    ASSERT_NEAR(wrap_angle_to_pi(-3. * kPI / 2.), 0.5 * kPI, 1e-6);
    // 360 degrees
    ASSERT_NEAR(wrap_angle_to_pi(2. * kPI), 0., 1e-6);
    // -360 degrees
    ASSERT_NEAR(wrap_angle_to_pi(-2. * kPI), 0., 1e-6);
    // 405 degrees = 360 + 45
    ASSERT_NEAR(wrap_angle_to_pi(2. * kPI + kPI / 4.), kPI / 4., 1e-6);
    // -405 degrees = -360 - 45
    ASSERT_NEAR(wrap_angle_to_pi(-2. * kPI - kPI / 4.), -kPI / 4., 1e-6);
    // 14 * 2 * pi + pi
    ASSERT_NEAR(wrap_angle_to_pi(29. * kPI), kPI, 1e-6);
    // -14 * 2 * pi - pi
    ASSERT_NEAR(wrap_angle_to_pi(-29. * kPI), -kPI, 1e-6);
    // 200. * pi + (pi / 6.)
    ASSERT_NEAR(wrap_angle_to_pi(200. * kPI + kPI / 6.), kPI / 6., 1e-6);
    // -200. * pi - (pi / 6.)
    ASSERT_NEAR(wrap_angle_to_pi(-200. * kPI - kPI / 6.), -kPI / 6., 1e-6);
}

}  // namespace openturbine::rigid_pendulum::tests
