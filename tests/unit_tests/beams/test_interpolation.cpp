#include <gtest/gtest.h>

#include "src/beams/interpolation.hpp"

namespace openturbine::tests {

constexpr auto tol = 1.e-15;

TEST(InterpolationTest, LinearInterpWeight) {
    std::vector<double> xs = {0., 1., 2.};
    std::vector<double> weights;
    std::vector<std::pair<double, std::vector<double>>> test_cases = {
        {-1.0, {1., 0., 0.}},   // Before first node
        {1.0, {0., 1., 0.}},    // Exactly on first node
        {0.5, {0.5, 0.5, 0.}},  // Between nodes 2 & 3
        {3.0, {0., 0., 1.}}     // After last node
    };
    for (const auto& [x, expected] : test_cases) {
        LinearInterpWeights(x, xs, weights);
        ASSERT_EQ(weights.size(), expected.size()) << "Failed at x = " << x;
        for (size_t i = 0; i < weights.size(); ++i) {
            EXPECT_NEAR(weights[i], expected[i], tol)
                << "Mismatch at x = " << x << ", index = " << i;
        }
    }
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_FirstOrder) {
    // 2 nodes at GLL points for first order polynomial
    std::vector<double> xs = {-1., 1.};
    std::vector<double> weights;

    // Test point at -1 (on first node)
    LagrangePolynomialInterpWeights(-1., xs, weights);
    ASSERT_EQ(weights.size(), 2);
    EXPECT_NEAR(weights[0], 1., tol);
    EXPECT_NEAR(weights[1], 0., tol);

    // Test point at 0 (between nodes 1 & 2)
    LagrangePolynomialInterpWeights(0., xs, weights);
    ASSERT_EQ(weights.size(), 2);
    EXPECT_NEAR(weights[0], 0.5, tol);
    EXPECT_NEAR(weights[1], 0.5, tol);

    // Test point at 1 (on last node)
    LagrangePolynomialInterpWeights(1., xs, weights);
    ASSERT_EQ(weights.size(), 2);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 1., tol);
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_SecondOrder) {
    // 3 nodes at GLL points for second order polynomial
    std::vector<double> xs = {-1., 0., 1.};
    std::vector<double> weights;

    // Test point at -1 (on first node)
    LagrangePolynomialInterpWeights(-1., xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 1., tol);
    EXPECT_NEAR(weights[1], 0., tol);
    EXPECT_NEAR(weights[2], 0., tol);

    // Test point at 0 (on second node)
    LagrangePolynomialInterpWeights(0., xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 1., tol);
    EXPECT_NEAR(weights[2], 0., tol);

    // Test point at 1 (on last node)
    LagrangePolynomialInterpWeights(1., xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 0., tol);
    EXPECT_NEAR(weights[2], 1., tol);
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_FourthOrder) {
    // 5 nodes at GLL points for fourth order polynomial
    std::vector<double> xs = {-1., -0.6546536707079771, 0., 0.6546536707079771, 1.};
    std::vector<double> weights;

    // Test point at -1 (on first node)
    LagrangePolynomialInterpWeights(-1., xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 1., tol);
    EXPECT_NEAR(weights[1], 0., tol);
    EXPECT_NEAR(weights[2], 0., tol);
    EXPECT_NEAR(weights[3], 0., tol);
    EXPECT_NEAR(weights[4], 0., tol);

    // Test point at -0.6546536707079771 (on second node)
    LagrangePolynomialInterpWeights(-0.6546536707079771, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 1., tol);
    EXPECT_NEAR(weights[2], 0., tol);
    EXPECT_NEAR(weights[3], 0., tol);
    EXPECT_NEAR(weights[4], 0., tol);

    // Test point at 0 (on third node)
    LagrangePolynomialInterpWeights(0., xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 0., tol);
    EXPECT_NEAR(weights[2], 1., tol);
    EXPECT_NEAR(weights[3], 0., tol);
    EXPECT_NEAR(weights[4], 0., tol);

    // Test point at 0.6546536707079771 (on fourth node)
    LagrangePolynomialInterpWeights(0.6546536707079771, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 0., tol);
    EXPECT_NEAR(weights[2], 0., tol);
    EXPECT_NEAR(weights[3], 1., tol);
    EXPECT_NEAR(weights[4], 0., tol);

    // Test point at 1 (on last node)
    LagrangePolynomialInterpWeights(1., xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 0., tol);
    EXPECT_NEAR(weights[2], 0., tol);
    EXPECT_NEAR(weights[3], 0., tol);
    EXPECT_NEAR(weights[4], 1., tol);

    // Test point at -0.8 (between nodes 1 & 2)
    LagrangePolynomialInterpWeights(-0.8, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0.2664, tol);
    EXPECT_NEAR(weights[1], 0.855336358376291, tol);
    EXPECT_NEAR(weights[2], -0.1776, tol);
    EXPECT_NEAR(weights[3], 0.0854636416237095, tol);
    EXPECT_NEAR(weights[4], -0.0296, tol);

    // Test point at 0.1 (between nodes 2 & 3)
    LagrangePolynomialInterpWeights(0.1, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0.0329625, tol);
    EXPECT_NEAR(weights[1], -0.1121093731918499, tol);
    EXPECT_NEAR(weights[2], 0.9669, tol);
    EXPECT_NEAR(weights[3], 0.1525343731918499, tol);
    EXPECT_NEAR(weights[4], -0.0402875, tol);

    // Test point at 0.4 (between nodes 3 & 4)
    LagrangePolynomialInterpWeights(0.4, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0.0564, tol);
    EXPECT_NEAR(weights[1], -0.1746924181056723, tol);
    EXPECT_NEAR(weights[2], 0.5263999999999998, tol);
    EXPECT_NEAR(weights[3], 0.7234924181056726, tol);
    EXPECT_NEAR(weights[4], -0.1316, tol);
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_NodesNotAtGLLPoints) {
    std::vector<double> xs = {0., 1., 2.};
    std::vector<double> weights;

    // Test point at x = 0.5 (between first two nodes)
    LagrangePolynomialInterpWeights(0.5, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.375, tol);
    EXPECT_NEAR(weights[1], 0.75, tol);
    EXPECT_NEAR(weights[2], -0.125, tol);

    // Test point exactly on middle node
    LagrangePolynomialInterpWeights(1., xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 1., tol);
    EXPECT_NEAR(weights[2], 0., tol);

    // Test point outside the node range
    LagrangePolynomialInterpWeights(-1.0, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 3., tol);
    EXPECT_NEAR(weights[1], -3., tol);
    EXPECT_NEAR(weights[2], 1., tol);
}

}  // namespace openturbine::tests
