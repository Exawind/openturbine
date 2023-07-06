#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/quaternion.h"

namespace openturbine::rigid_pendulum::tests {

using namespace openturbine::rigid_pendulum;

TEST(QuaternionTest, DefaultConstructor) {
    Quaternion q;
    std::array<double, 4> expected = {0., 0., 0., 0.};

    ASSERT_EQ(q.values(), expected);
}

TEST(QuaternionTest, ArrayConstructor) {
    std::array<double, 4> values = {1., 2., 3., 4.};
    Quaternion q(values);

    ASSERT_EQ(q.values(), values);
}

TEST(QuaternionTest, ScalarVectorConstructor) {
    double scalar = 1.;
    std::array<double, 3> vector = {2., 3., 4.};
    Quaternion q(scalar, vector);
    std::array<double, 4> expected = {1., 2., 3., 4.};

    ASSERT_EQ(q.values(), expected);
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

    ASSERT_DOUBLE_EQ(q.length(), std::sqrt(30.0));
}

}  // namespace openturbine::rigid_pendulum::tests
