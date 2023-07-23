#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/utilities.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(MathUtilitiesTest, CloseTo_Set1) {
    ASSERT_TRUE(close_to(1., 1.));
}

TEST(MathUtilitiesTest, CloseTo_Set2) {
    ASSERT_TRUE(close_to(1., 1. + kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set3) {
    ASSERT_TRUE(close_to(1., 1. - kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set4) {
    ASSERT_FALSE(close_to(1., 1. + kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set5) {
    ASSERT_FALSE(close_to(1., 1. - kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set6) {
    ASSERT_TRUE(close_to(kTOLERANCE / 10., kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set7) {
    ASSERT_TRUE(close_to(-1., -1.));
}

TEST(MathUtilitiesTest, CloseTo_Set8) {
    ASSERT_TRUE(close_to(-1., -1. + kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set9) {
    ASSERT_TRUE(close_to(-1., -1. - kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set10) {
    ASSERT_FALSE(close_to(-1., -1. + kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set11) {
    ASSERT_FALSE(close_to(-1., -1. - kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set12) {
    ASSERT_TRUE(close_to(-1e-7, -kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set13) {
    ASSERT_FALSE(close_to(1., -1.));
}

TEST(MathUtilitiesTest, CloseTo_Set14) {
    ASSERT_FALSE(close_to(-1., 1.));
}

TEST(MathUtilitiesTest, CloseTo_Set15) {
    ASSERT_FALSE(close_to(1., -1. + kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set16) {
    ASSERT_FALSE(close_to(-1., 1. + kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set17) {
    ASSERT_FALSE(close_to(1., -1. - kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set18) {
    ASSERT_FALSE(close_to(-1., 1. - kTOLERANCE / 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set19) {
    ASSERT_FALSE(close_to(1., -1. + kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set20) {
    ASSERT_FALSE(close_to(-1., 1. + kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set21) {
    ASSERT_FALSE(close_to(1., -1. - kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, CloseTo_Set22) {
    ASSERT_FALSE(close_to(-1., 1. - kTOLERANCE * 10.));
}

TEST(MathUtilitiesTest, WrapAngleToPi_ZeroDegree) {
    // 0 degree
    auto angle = 0.;
    auto expected = 0.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_FortyFiveDegrees) {
    // 45 degrees
    auto angle = kPI / 4.;
    auto expected = kPI / 4.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeFortyFiveDegrees) {
    // -45 degrees
    auto angle = -kPI / 4.;
    auto expected = -kPI / 4.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NinetyDegrees) {
    // 90 degrees
    auto angle = kPI / 2.;
    auto expected = kPI / 2.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeNinetyDegrees) {
    // -90 degrees
    auto angle = -kPI / 2.;
    auto expected = -kPI / 2.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_OneHundredThirtyFiveDegrees) {
    // 135 degrees
    auto angle = kPI / 2. + kPI / 4.;
    auto expected = 0.75 * kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeOneHundredThirtyFiveDegrees) {
    // -135 degrees
    auto angle = -kPI / 2. - kPI / 4.;
    auto expected = -0.75 * kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_OneHundredEightyDegrees) {
    // 180 degrees
    auto angle = kPI;
    auto expected = kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeOneHundredEightyDegrees) {
    // -180 degrees
    auto angle = -kPI;
    auto expected = -kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_TwoHundredTwentyFiveDegrees) {
    // 225 degrees
    auto angle = kPI + kPI / 4.;
    auto expected = -0.75 * kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeTwoHundredTwentyFiveDegrees) {
    // -225 degrees
    auto angle = -kPI - kPI / 4.;
    auto expected = 0.75 * kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_TwoHundredSeventyDegrees) {
    // 270 degrees
    auto angle = 3. * kPI / 2.;
    auto expected = -0.5 * kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeTwoHundredSeventyDegrees) {
    // -270 degrees
    auto angle = -3. * kPI / 2.;
    auto expected = 0.5 * kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_ThreeHundredSixtyDegrees) {
    // 360 degrees
    auto angle = 2. * kPI;
    auto expected = 0.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeThreeHundredSixtyDegrees) {
    // -360 degrees
    auto angle = -2. * kPI;
    auto expected = 0.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_FourHundredFiveDegrees) {
    // 405 degrees = 360 + 45
    auto angle = 2. * kPI + kPI / 4.;
    auto expected = kPI / 4.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeFourHundredFiveDegrees) {
    // -405 degrees = -360 - 45
    auto angle = -2. * kPI - kPI / 4.;
    auto expected = -kPI / 4.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_FiveThousandsTwoHundredTwentyDegrees) {
    // 14 * 2 * pi + pi
    auto angle = 29. * kPI;
    auto expected = kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeFiveThousandsTwoHundredTwentyDegrees) {
    // -14 * 2 * pi - pi
    auto angle = -29. * kPI;
    auto expected = -kPI;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_ThirtySixThousandThirtyDegrees) {
    // 200. * pi + (pi / 6.)
    auto angle = 200. * kPI + kPI / 6.;
    auto expected = kPI / 6.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, WrapAngleToPi_NegativeThirtySixThousandThirtyDegrees) {
    // -200. * pi - (pi / 6.)
    auto angle = -200. * kPI - kPI / 6.;
    auto expected = -kPI / 6.;

    ASSERT_NEAR(wrap_angle_to_pi(angle), expected, 1e-6);
}

TEST(MathUtilitiesTest, CreateKokkosView1DFromGivenVector) {
    std::vector<double> values = {1., 2., 3., 4., 5.};
    auto vector = create_vector(values);

    expect_kokkos_view_1D_equal(vector, values);
}

TEST(MathUtilitiesTest, CreateKokkosView2DFromGivenMatrix) {
    std::vector<std::vector<double>> values = {
        {1., 2., 3., 4., 5.},
        {6., 7., 8., 9., 10.},
        {11., 12., 13., 14., 15.},
        {16., 17., 18., 19., 20.},
        {21., 22., 23., 24., 25.}};
    auto matrix = create_matrix(values);

    expect_kokkos_view_2D_equal(matrix, values);
}

}  // namespace openturbine::rigid_pendulum::tests
