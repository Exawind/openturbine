#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/quaternion.h"

namespace openturbine::rigid_pendulum::tests {

using namespace openturbine::rigid_pendulum;

TEST(QuaternionTest, DefaultConstructor) {
    Quaternion q;
    std::array<double, 4> expected = {0., 0., 0., 0.};

    ASSERT_EQ(q.GetComponents(), expected);
}

TEST(QuaternionTest, ArrayConstructor) {
    std::array<double, 4> values = {1., 2., 3., 4.};
    Quaternion q(values);

    ASSERT_EQ(q.GetComponents(), values);
}

TEST(QuaternionTest, ScalarVectorConstructor) {
    double scalar = 1.;
    std::array<double, 3> vector = {2., 3., 4.};
    Quaternion q(scalar, vector);
    std::array<double, 4> expected = {1., 2., 3., 4.};

    ASSERT_EQ(q.GetComponents(), expected);
}

TEST(QuaternionTest, IndexOperator) {
    Quaternion q(std::array{1., 2., 3., 4.});

    ASSERT_EQ(q[0], 1.);
    ASSERT_EQ(q[1], 2.);
    ASSERT_EQ(q[2], 3.);
    ASSERT_EQ(q[3], 4.);

    ASSERT_THROW(q[4], std::out_of_range);
}

TEST(QuaternionTest, Length) {
    Quaternion q(std::array{1.0, 2.0, 3.0, 4.0});

    ASSERT_DOUBLE_EQ(q.Length(), std::sqrt(30.0));
}

TEST(QuaternionTest, GetScalarComponent) {
    Quaternion q(std::array{1., 2., 3., 4.});

    ASSERT_EQ(q.GetScalarComponent(), 1.);
}

TEST(QuaternionTest, GetVectorComponent) {
    Quaternion q(std::array{1., 2., 3., 4.});
    std::array<double, 3> expected = {2., 3., 4.};

    ASSERT_EQ(q.GetVectorComponent(), expected);
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

TEST(QuaternionTest, ExpectNonUnitQuaternion) {
    Quaternion q(std::array{1., 2., 3., 4.});

    ASSERT_FALSE(q.IsUnitQuaternion());
}

TEST(QuaternionTest, ExpectUnitQuaternion) {
    double l = std::sqrt(30.);
    Quaternion q(std::array{1. / l, 2. / l, 3. / l, 4. / l});

    ASSERT_TRUE(q.IsUnitQuaternion());
}

TEST(QuaternionTest, GetUnitQuaternion) {
    Quaternion q(std::array{1., 2., 3., 4.});
    Quaternion expected(std::array{
        1. / std::sqrt(30.), 2. / std::sqrt(30.), 3. / std::sqrt(30.), 4. / std::sqrt(30.)});

    ASSERT_EQ(q.GetUnitQuaternion().GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, AdditionOfTwoQuaternions) {
    Quaternion q1(std::array{1., 2., 3., 4.});
    Quaternion q2(std::array{5., 6., 7., 8.});
    Quaternion expected(std::array{6., 8., 10., 12.});

    ASSERT_EQ((q1 + q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, AdditionOfThreeQuaternions) {
    Quaternion q1(std::array{1., 2., 3., 4.});
    Quaternion q2(std::array{5., 6., 7., 8.});
    Quaternion q3(std::array{9., 10., 11., 12.});
    Quaternion expected(std::array{15., 18., 21., 24.});

    ASSERT_EQ((q1 + q2 + q3).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, SubtractionOfTwoQuaternions) {
    Quaternion q1(std::array{1., 2., 3., 4.});
    Quaternion q2(std::array{5., 6., 7., 8.});
    Quaternion expected(std::array{-4., -4., -4., -4.});

    ASSERT_EQ((q1 - q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, AdditionAndSubtractionOfThreeQuaternions) {
    Quaternion q1(std::array{1., 2., 3., 4.});
    Quaternion q2(std::array{5., 6., 7., 8.});
    Quaternion q3(std::array{9., 10., 11., 12.});
    Quaternion expected(std::array{-3., -2., -1., 0.});

    ASSERT_EQ((q1 + q2 - q3).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_1) {
    Quaternion q1(std::array{3., 1., -2., 1.});
    Quaternion q2(std::array{2., -1., 2., 3.});
    Quaternion expected(std::array{8., -9., -2., 11.});

    ASSERT_EQ((q1 * q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, MultiplicationOfTwoQuaternions_2) {
    Quaternion q1(std::array{1., 2., 3., 4.});
    Quaternion q2(std::array{5., 6., 7., 8.});
    Quaternion expected(std::array{-60., 12., 30., 24.});

    ASSERT_EQ((q1 * q2).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, MultiplicationOfQuaternionAndScalar) {
    Quaternion q(std::array{1., 2., 3., 4.});
    Quaternion expected(std::array{2., 4., 6., 8.});

    ASSERT_EQ((q * 2.).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, DivisionOfQuaternionAndScalar) {
    Quaternion q(std::array{1., 2., 3., 4.});
    Quaternion expected(std::array{0.5, 1., 1.5, 2.});

    ASSERT_EQ((q / 2.).GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, GetConjugate) {
    Quaternion q(std::array{1., 2., 3., 4.});
    Quaternion expected(std::array{1., -2., -3., -4.});

    ASSERT_EQ(q.GetConjugate().GetComponents(), expected.GetComponents());
}

TEST(QuaternionTest, GetInverse) {
    Quaternion q(std::array{1., 2., 3., 4.});
    Quaternion expected(std::array{1. / 30., -2. / 30., -3. / 30., -4. / 30.});

    ASSERT_EQ(q.GetInverse().GetComponents(), expected.GetComponents());
    ASSERT_TRUE((q * q.GetInverse()).IsUnitQuaternion());
}

Quaternion quaternion_from_rotation_vector(const std::array<double, 3>& rotation_vector) {
    double angle = std::sqrt(
        rotation_vector[0] * rotation_vector[0] + rotation_vector[1] * rotation_vector[1] +
        rotation_vector[2] * rotation_vector[2]
    );

    if (close_to(angle, 0.)) {
        return Quaternion(1.0, 0.0, 0.0, 0.0);
    }

    double sin_angle = std::sin(angle / 2.0);
    double cos_angle = std::cos(angle / 2.0);
    auto factor = sin_angle / angle;

    return Quaternion(
        cos_angle, rotation_vector[0] * factor, rotation_vector[1] * factor,
        rotation_vector[2] * factor
    );
}

TEST(QuaternionTest, GetQuaternionFromRotationVector_Set1) {
    std::array<double, 3> rotation_vector{1., 2., 3.};
    Quaternion expected(std::array{-0.295551, 0.255322, 0.510644, 0.765966});

    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[0],
        expected.GetComponents()[0], 1e-6
    );
    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[1],
        expected.GetComponents()[1], 1e-6
    );
    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[2],
        expected.GetComponents()[2], 1e-6
    );
    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[3],
        expected.GetComponents()[3], 1e-6
    );
}

TEST(QuaternionTest, GetRotationVectorFromQuaternion_Set1) {
    Quaternion q(std::array{-0.295551, 0.255322, 0.510644, 0.765966});
    std::array<double, 3> expected{1., 2., 3.};

    ASSERT_NEAR(rotation_vector_from_quaternion(q)[0], expected[0], 1e-6);
    ASSERT_NEAR(rotation_vector_from_quaternion(q)[1], expected[1], 1e-6);
    ASSERT_NEAR(rotation_vector_from_quaternion(q)[2], expected[2], 1e-6);
}

TEST(QuaternionTest, GetQuaternionFromRotationVector_Set2) {
    std::array<double, 3> rotation_vector{0.0, 0.0, 1.570796};
    Quaternion expected(std::array{0.707107, 0., 0., 0.707107});

    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[0],
        expected.GetComponents()[0], 1e-6
    );
    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[1],
        expected.GetComponents()[1], 1e-6
    );
    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[2],
        expected.GetComponents()[2], 1e-6
    );
    ASSERT_NEAR(
        quaternion_from_rotation_vector(rotation_vector).GetComponents()[3],
        expected.GetComponents()[3], 1e-6
    );
}

TEST(QuaternionTest, GetRotationVectorFromQuaternion_Set2) {
    Quaternion q(std::array{0.707107, 0., 0., 0.707107});
    std::array<double, 3> expected{0.0, 0.0, 1.570796};

    ASSERT_NEAR(rotation_vector_from_quaternion(q)[0], expected[0], 1e-6);
    ASSERT_NEAR(rotation_vector_from_quaternion(q)[1], expected[1], 1e-6);
    ASSERT_NEAR(rotation_vector_from_quaternion(q)[2], expected[2], 1e-6);
}

TEST(QuaternionTest, GetQuaternionFromNullRotationVector) {
    std::array<double, 3> rotation_vector{0., 0., 0.};
    Quaternion expected(std::array{1., 0., 0., 0.});

    ASSERT_EQ(
        quaternion_from_rotation_vector(rotation_vector).GetComponents(), expected.GetComponents()
    );
}

TEST(QuaternionTest, GetRotationVectorFromNullQuaternion) {
    Quaternion q(std::array{0., 0., 0., 0.});
    std::array<double, 3> expected{0., 0., 0.};

    ASSERT_EQ(rotation_vector_from_quaternion(q), expected);
}

}  // namespace openturbine::rigid_pendulum::tests
