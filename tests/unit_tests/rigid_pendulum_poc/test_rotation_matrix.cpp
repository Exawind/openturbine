#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/rotation_matrix.h"

TEST(RotationMatrixTest, DefaultToZero) {
    openturbine::rigid_pendulum::RotationMatrix matrix;
    ASSERT_EQ(matrix(0, 0), 0.);
    ASSERT_EQ(matrix(0, 1), 0.);
    ASSERT_EQ(matrix(0, 2), 0.);

    ASSERT_EQ(matrix(1, 0), 0.);
    ASSERT_EQ(matrix(1, 1), 0.);
    ASSERT_EQ(matrix(1, 2), 0.);

    ASSERT_EQ(matrix(2, 0), 0.);
    ASSERT_EQ(matrix(2, 1), 0.);
    ASSERT_EQ(matrix(2, 2), 0.);
}

TEST(RotationMatrixTest, CreateFromComponents) {
    openturbine::rigid_pendulum::RotationMatrix matrix(1., 2., 3., 4., 5., 6., 7., 8., 9.);
    ASSERT_EQ(matrix(0, 0), 1.);
    ASSERT_EQ(matrix(0, 1), 2.);
    ASSERT_EQ(matrix(0, 2), 3.);

    ASSERT_EQ(matrix(1, 0), 4.);
    ASSERT_EQ(matrix(1, 1), 5.);
    ASSERT_EQ(matrix(1, 2), 6.);

    ASSERT_EQ(matrix(2, 0), 7.);
    ASSERT_EQ(matrix(2, 1), 8.);
    ASSERT_EQ(matrix(2, 2), 9.);
}