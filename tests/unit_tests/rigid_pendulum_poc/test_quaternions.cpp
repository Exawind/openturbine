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

}  // namespace openturbine::rigid_pendulum::tests
