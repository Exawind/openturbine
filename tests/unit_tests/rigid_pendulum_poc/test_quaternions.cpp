#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/quaternion.h"

namespace openturbine::rigid_pendulum::tests {

TEST(QuaternionTest, DefaultConstructor) {
    Quaternion q;
    std::tuple<double, double, double, double> expected = {0., 0., 0., 0.};

    ASSERT_EQ(q.GetComponents(), expected);
}

TEST(QuaternionTest, ConstructorWithProvidedComponents) {
    Quaternion q(1., 2., 3., 4.);
    std::tuple<double, double, double, double> expected = {1., 2., 3., 4.};

    ASSERT_EQ(q.GetComponents(), expected);
}

TEST(QuaternionTest, GetIndividualComponents) {
    Quaternion q(1., 2., 3., 4.);

    ASSERT_EQ(q.GetScalarComponent(), 1.);
    ASSERT_EQ(q.GetXComponent(), 2.);
    ASSERT_EQ(q.GetYComponent(), 3.);
    ASSERT_EQ(q.GetZComponent(), 4.);
}

TEST(QuaternionTest, Length) {
    Quaternion q(1., 2., 3., 4.);

    ASSERT_EQ(q.Length(), std::sqrt(30.));
}

TEST(QuaternionTest, CloseTo) {
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

TEST(QuaternionTest, AdditionOfTwoQuaternions) {
    Quaternion q1(1., 2., 3., 4.);
    Quaternion q2(5., 6., 7., 8.);
    Quaternion expected(6., 8., 10., 12.);

    ASSERT_EQ((q1 + q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, AdditionOfThreeQuaternions) {
    Quaternion q1(1., 2., 3., 4.);
    Quaternion q2(5., 6., 7., 8.);
    Quaternion q3(9., 10., 11., 12.);
    Quaternion expected(15., 18., 21., 24.);

    ASSERT_EQ((q1 + q2 + q3).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, SubtractionOfTwoQuaternions) {
    Quaternion q1(1., 2., 3., 4.);
    Quaternion q2(5., 6., 7., 8.);
    Quaternion expected(-4., -4., -4., -4.);

    ASSERT_EQ((q1 - q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, AdditionAndSubtractionOfThreeQuaternions) {
    Quaternion q1(1., 2., 3., 4.);
    Quaternion q2(5., 6., 7., 8.);
    Quaternion q3(9., 10., 11., 12.);
    Quaternion expected(-3., -2., -1., 0.);

    ASSERT_EQ((q1 + q2 - q3).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set1) {
    Quaternion q1(3., 1., -2., 1.);
    Quaternion q2(2., -1., 2., 3.);
    Quaternion expected(8., -9., -2., 11.);

    ASSERT_EQ((q1 * q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_Set2) {
    Quaternion q1(1., 2., 3., 4.);
    Quaternion q2(5., 6., 7., 8.);
    Quaternion expected(-60., 12., 30., 24.);

    ASSERT_EQ((q1 * q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, MultiplicationOfQuaternionAndScalar) {
    Quaternion q(1., 2., 3., 4.);
    Quaternion expected(2., 4., 6., 8.);

    ASSERT_EQ((q * 2.).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, DivisionOfQuaternionAndScalar) {
    Quaternion q(1., 2., 3., 4.);
    Quaternion expected(0.5, 1., 1.5, 2.);

    ASSERT_EQ((q / 2.).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, ExpectNonUnitQuaternion) {
    Quaternion q(1., 2., 3., 4.);

    ASSERT_FALSE(q.IsUnitQuaternion());
}

TEST(QuaternionTest, ExpectUnitQuaternion) {
    double l = std::sqrt(30.);
    Quaternion q(1. / l, 2. / l, 3. / l, 4. / l);

    ASSERT_TRUE(q.IsUnitQuaternion());
}

TEST(QuaternionTest, GetUnitQuaternion) {
    auto sqrt_30 = std::sqrt(30.);
    Quaternion q(1., 2., 3., 4.);
    Quaternion expected(1. / sqrt_30, 2. / sqrt_30, 3. / sqrt_30, 4. / sqrt_30);

    ASSERT_EQ(q.GetUnitQuaternion().GetComponents(), expected.GetComponents());
    ASSERT_TRUE(expected.IsUnitQuaternion());
}

TEST(QuaternionTest, GetConjugate) {
    Quaternion q(1., 2., 3., 4.);
    Quaternion expected(1., -2., -3., -4.);

    ASSERT_EQ(q.GetConjugate().GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, GetInverse) {
    Quaternion q(1., 2., 3., 4.);
    Quaternion expected(1. / 30., -2. / 30., -3. / 30., -4. / 30.);

    ASSERT_EQ(q.GetInverse().GetComponents(), expected.GetComponents());

    auto q_inv = q.GetInverse();
    ASSERT_TRUE((q * q_inv).IsUnitQuaternion());
}

TEST(QuaternionTest, GetQuaternionFromRotationVector_Set1) {
    std::tuple<double, double, double> rotation_vector{1., 2., 3.};
    auto q = quaternion_from_rotation_vector(rotation_vector);

    // We will use the following quaternion as input in the log conversion
    Quaternion expected(-0.295551, 0.255322, 0.510644, 0.765966);

    ASSERT_NEAR(q.GetScalarComponent(), expected.GetScalarComponent(), 1e-6);
    ASSERT_NEAR(q.GetXComponent(), expected.GetXComponent(), 1e-6);
    ASSERT_NEAR(q.GetYComponent(), expected.GetYComponent(), 1e-6);
    ASSERT_NEAR(q.GetZComponent(), expected.GetZComponent(), 1e-6);
}

TEST(QuaternionTest, GetRotationVectorFromQuaternion_Set1) {
    Quaternion q(-0.295551, 0.255322, 0.510644, 0.765966);
    auto v = rotation_vector_from_quaternion(q);

    // We expect the rotation vector to be same as provided in above exp conversion
    // i.e. {1., 2., 3.}
    std::tuple<double, double, double> expected{1., 2., 3.};

    ASSERT_NEAR(std::get<0>(v), std::get<0>(expected), 1e-6);
    ASSERT_NEAR(std::get<1>(v), std::get<1>(expected), 1e-6);
    ASSERT_NEAR(std::get<2>(v), std::get<2>(expected), 1e-6);
}

TEST(QuaternionTest, GetQuaternionFromRotationVector_Set2) {
    std::tuple<double, double, double> rotation_vector{0., 0., 1.570796};
    auto q = quaternion_from_rotation_vector(rotation_vector);

    // We will use the following quaternion as input in the log conversion
    Quaternion expected(0.707107, 0., 0., 0.707107);

    ASSERT_NEAR(q.GetScalarComponent(), expected.GetScalarComponent(), 1e-6);
    ASSERT_NEAR(q.GetXComponent(), expected.GetXComponent(), 1e-6);
    ASSERT_NEAR(q.GetYComponent(), expected.GetYComponent(), 1e-6);
    ASSERT_NEAR(q.GetZComponent(), expected.GetZComponent(), 1e-6);
}

TEST(QuaternionTest, GetRotationVectorFromQuaternion_Set2) {
    Quaternion q(0.707107, 0., 0., 0.707107);
    auto v = rotation_vector_from_quaternion(q);

    // We expect the rotation vector to be same as provided in above exp conversion
    // i.e. {0., 0., 1.570796}
    std::tuple<double, double, double> expected{0., 0., 1.570796};

    ASSERT_NEAR(std::get<0>(v), std::get<0>(expected), 1e-6);
    ASSERT_NEAR(std::get<1>(v), std::get<1>(expected), 1e-6);
    ASSERT_NEAR(std::get<2>(v), std::get<2>(expected), 1e-6);
}

TEST(QuaternionTest, GetQuaternionFromNullRotationVector) {
    std::tuple<double, double, double> rotation_vector{0., 0., 0.};
    auto q = quaternion_from_rotation_vector(rotation_vector);
    Quaternion expected(1., 0., 0., 0.);

    ASSERT_NEAR(q.GetScalarComponent(), expected.GetScalarComponent(), 1e-6);
    ASSERT_NEAR(q.GetXComponent(), expected.GetXComponent(), 1e-6);
    ASSERT_NEAR(q.GetYComponent(), expected.GetYComponent(), 1e-6);
    ASSERT_NEAR(q.GetZComponent(), expected.GetZComponent(), 1e-6);
}

TEST(QuaternionTest, GetRotationVectorFromNullQuaternion) {
    Quaternion q(1., 0., 0., 0.);
    auto v = rotation_vector_from_quaternion(q);
    std::tuple<double, double, double> expected{0., 0., 0.};

    ASSERT_NEAR(std::get<0>(v), std::get<0>(expected), 1e-6);
    ASSERT_NEAR(std::get<1>(v), std::get<1>(expected), 1e-6);
    ASSERT_NEAR(std::get<2>(v), std::get<2>(expected), 1e-6);
}

TEST(QuaternionTest, WrapAngleToPi) {
    auto angle = 0.;  // 0 degrees
    auto wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, 0., 1e-6);

    angle = kPI / 2.;  // 90 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, 0.5 * kPI, 1e-6);

    angle = -kPI / 2.;  // -90 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, -0.5 * kPI, 1e-6);

    angle = kPI / 2. + kPI / 4.;  // 135 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, 0.75 * kPI, 1e-6);

    angle = -kPI / 2. - kPI / 4.;  // -135 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, -0.75 * kPI, 1e-6);

    angle = kPI;  // 180 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, kPI, 1e-6);

    angle = -kPI;
    wrapped_angle = wrap_angle_to_pi(angle);  // -180 degrees
    ASSERT_NEAR(wrapped_angle, -kPI, 1e-6);

    angle = 3. * kPI / 2.;  // 270 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, -0.5 * kPI, 1e-6);

    angle = -3. * kPI / 2.;  // -270 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, 0.5 * kPI, 1e-6);

    angle = 2. * kPI;  // 360 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, 0., 1e-6);

    angle = -2. * kPI;  // -360 degrees
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, 0., 1e-6);

    angle = 2. * kPI + kPI / 4.;  // 405 degrees = 360 + 45
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, kPI / 4., 1e-6);

    angle = -2. * kPI - kPI / 4.;  // -405 degrees = -360 - 45
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, -kPI / 4., 1e-6);

    angle = 29. * kPI;  // 14 * 2 * kPI + kPI
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, kPI, 1e-6);

    angle = -29. * kPI;  // -14 * 2 * kPI - kPI
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, -kPI, 1e-6);

    angle = 200. * kPI + kPI / 6.;
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, kPI / 6., 1e-6);

    angle = -200. * kPI - kPI / 6.;
    wrapped_angle = wrap_angle_to_pi(angle);
    ASSERT_NEAR(wrapped_angle, -kPI / 6., 1e-6);
}

}  // namespace openturbine::rigid_pendulum::tests
