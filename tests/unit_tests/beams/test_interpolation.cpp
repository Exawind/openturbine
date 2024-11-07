#include <gtest/gtest.h>

#include "src/beams/interpolation.hpp"

namespace openturbine::tests {

constexpr auto tol = 1.e-15;

TEST(InterpolationTest, LinearInterpWeight_BetweenNodes) {
    std::vector<double> xs = {0.0, 1.0, 2.0};
    std::vector<double> weights;

    // Test point at x = 0.5 (between first two nodes)
    LinearInterpWeights(0.5, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.5, tol);
    EXPECT_NEAR(weights[1], 0.5, tol);
    EXPECT_NEAR(weights[2], 0.0, tol);
}

TEST(InterpolationTest, LinearInterpWeight_ExactlyOnNode) {
    std::vector<double> xs = {0.0, 1.0, 2.0};
    std::vector<double> weights;

    // Test point exactly on middle node
    LinearInterpWeights(1.0, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.0, tol);
    EXPECT_NEAR(weights[1], 1.0, tol);
    EXPECT_NEAR(weights[2], 0.0, tol);
}

TEST(InterpolationTest, LinearInterpWeight_BeforeFirstNode) {
    std::vector<double> xs = {0.0, 1.0, 2.0};
    std::vector<double> weights;

    // Test point before first node
    LinearInterpWeights(-1.0, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 1.0, tol);
    EXPECT_NEAR(weights[1], 0.0, tol);
    EXPECT_NEAR(weights[2], 0.0, tol);
}

TEST(InterpolationTest, LinearInterpWeight_AfterLastNode) {
    std::vector<double> xs = {0.0, 1.0, 2.0};
    std::vector<double> weights;

    // Test point after last node
    LinearInterpWeights(3.0, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.0, tol);
    EXPECT_NEAR(weights[1], 0.0, tol);
    EXPECT_NEAR(weights[2], 1.0, tol);
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_BetweenNodes) {
    std::vector<double> xs = {0.0, 1.0, 2.0};
    std::vector<double> weights;

    // Test point at x = 0.5 (between first two nodes)
    LagrangePolynomialInterpWeights(0.5, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.375, tol);
    EXPECT_NEAR(weights[1], 0.75, tol);
    EXPECT_NEAR(weights[2], -0.125, tol);
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_ExactlyOnNode) {
    std::vector<double> xs = {0.0, 1.0, 2.0};
    std::vector<double> weights;

    // Test point exactly on middle node
    LagrangePolynomialInterpWeights(1.0, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.0, tol);
    EXPECT_NEAR(weights[1], 1.0, tol);
    EXPECT_NEAR(weights[2], 0.0, tol);
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_OutsideNodes) {
    std::vector<double> xs = {0.0, 1.0, 2.0};
    std::vector<double> weights;

    // Test point outside the node range
    LagrangePolynomialInterpWeights(-1.0, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 3.0, tol);
    EXPECT_NEAR(weights[1], -3.0, tol);
    EXPECT_NEAR(weights[2], 1.0, tol);
}

}  // namespace openturbine::tests
