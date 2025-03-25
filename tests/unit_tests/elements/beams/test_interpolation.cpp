#include <gtest/gtest.h>

#include "elements/beams/interpolation.hpp"

namespace openturbine::tests {

constexpr auto tol = 1.e-12;

TEST(InterpolationTest, LinearInterpWeight) {
    const std::vector<double> xs = {0., 1., 2.};
    std::vector<double> weights;
    const std::vector<std::pair<double, std::vector<double>>> test_cases = {
        {-1., {1., 0., 0.}},    // Before first node
        {1., {0., 1., 0.}},     // Exactly on first node
        {0.5, {0.5, 0.5, 0.}},  // Between nodes 2 & 3
        {3., {0., 0., 1.}}      // After last node
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
    const std::vector<double> xs = {-1., 1.};
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
    const std::vector<double> xs = {-1., 0., 1.};
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

TEST(InterpolationTest, LagrangePolynomialInterpWeight_SecondOrder_AtSpecifiedQuadraturePoints) {
    // 3 nodes at GLL points for second order polynomial
    const std::vector<double> xs = {-1., 0., 1.};
    std::vector<double> weights;

    // We need to calculate shape function weights at following 7 Gauss quadrature pts:
    // -0.9491079123427585,-0.7415311855993945,-0.4058451513773972,0.,0.4058451513773972,0.7415311855993945,0.9491079123427585

    // Test point at -0.9491079123427585 (on first QP)
    LagrangePolynomialInterpWeights(-0.9491079123427585, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.924956870807, tol);
    EXPECT_NEAR(weights[1], 0.0991941707284, tol);
    EXPECT_NEAR(weights[2], -0.0241510415356, tol);

    // Test point at -0.7415311855993945 (on second QP)
    LagrangePolynomialInterpWeights(-0.7415311855993945, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.645699842408, tol);
    EXPECT_NEAR(weights[1], 0.450131500784, tol);
    EXPECT_NEAR(weights[2], -0.0958313431915, tol);

    // Test point at -0.4058451513773972 (on third QP)
    LagrangePolynomialInterpWeights(-0.4058451513773972, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0.285277719137, tol);
    EXPECT_NEAR(weights[1], 0.835289713103, tol);
    EXPECT_NEAR(weights[2], -0.12056743224, tol);

    // Test point at 0. (on fourth QP)
    LagrangePolynomialInterpWeights(0., xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], 0., tol);
    EXPECT_NEAR(weights[1], 1., tol);
    EXPECT_NEAR(weights[2], 0., tol);

    // Test point at 0.4058451513773972 (on fifth QP)
    LagrangePolynomialInterpWeights(0.4058451513773972, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], -0.12056743224, tol);
    EXPECT_NEAR(weights[1], 0.835289713103, tol);
    EXPECT_NEAR(weights[2], 0.285277719137, tol);

    // Test point at 0.7415311855993945 (on sixth QP)
    LagrangePolynomialInterpWeights(0.7415311855993945, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], -0.0958313431915, tol);
    EXPECT_NEAR(weights[1], 0.450131500784, tol);
    EXPECT_NEAR(weights[2], 0.645699842408, tol);

    // Test point at 0.9491079123427585 (on seventh QP)
    LagrangePolynomialInterpWeights(0.9491079123427585, xs, weights);
    ASSERT_EQ(weights.size(), 3);
    EXPECT_NEAR(weights[0], -0.0241510415356, tol);
    EXPECT_NEAR(weights[1], 0.0991941707284, tol);
    EXPECT_NEAR(weights[2], 0.924956870807, tol);
}

TEST(InterpolationTest, LagrangePolynomialInterpWeight_FourthOrder) {
    // 5 nodes at GLL points for fourth order polynomial
    const std::vector<double> xs = {-1., -0.6546536707079771, 0., 0.6546536707079771, 1.};
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

    // Test point at 0.1 (between nodes 3 & 4)
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
    const std::vector<double> xs = {0., 1., 2.};
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

TEST(InterpolationTest, LagrangePolynomialDerivWeights_FirstOrder) {
    // 2 nodes at GLL points for first order polynomial
    const std::vector<double> xs = {-1., 1.};
    std::vector<double> weights;

    // Test point at x = -1 (on first node)
    LagrangePolynomialDerivWeights(-1., xs, weights);
    ASSERT_EQ(weights.size(), 2);
    EXPECT_NEAR(weights[0], -0.5, tol);
    EXPECT_NEAR(weights[1], 0.5, tol);

    // Test point at x = 1 (on last node)
    LagrangePolynomialDerivWeights(1., xs, weights);
    ASSERT_EQ(weights.size(), 2);
    EXPECT_NEAR(weights[0], -0.5, tol);
    EXPECT_NEAR(weights[1], 0.5, tol);
}

TEST(InterpolationTest, LagrangePolynomialDerivWeight_SecondOrder_AtSpecifiedQuadraturePoints) {
    // 3 nodes at GLL points for second order polynomial
    const std::vector<double> xs = {-1., 0., 1.};
    std::vector<double> deriv_weights;

    // We need to calculate shape function derivative weights at following 7 Gauss quadrature pts:
    // -0.9491079123427585,-0.7415311855993945,-0.4058451513773972,0.,0.4058451513773972,0.7415311855993945,0.9491079123427585

    // Test point at -0.9491079123427585 (on first QP)
    LagrangePolynomialDerivWeights(-0.9491079123427585, xs, deriv_weights);
    ASSERT_EQ(deriv_weights.size(), 3);
    EXPECT_NEAR(deriv_weights[0], -1.449107912343, tol);
    EXPECT_NEAR(deriv_weights[1], 1.898215824686, tol);
    EXPECT_NEAR(deriv_weights[2], -0.4491079123428, tol);

    // Test point at -0.7415311855993945 (on second QP)
    LagrangePolynomialDerivWeights(-0.7415311855993945, xs, deriv_weights);
    ASSERT_EQ(deriv_weights.size(), 3);
    EXPECT_NEAR(deriv_weights[0], -1.241531185599, tol);
    EXPECT_NEAR(deriv_weights[1], 1.483062371199, tol);
    EXPECT_NEAR(deriv_weights[2], -0.2415311855994, tol);

    // Test point at -0.4058451513773972 (on third QP)
    LagrangePolynomialDerivWeights(-0.4058451513773972, xs, deriv_weights);
    ASSERT_EQ(deriv_weights.size(), 3);
    EXPECT_NEAR(deriv_weights[0], -0.9058451513774, tol);
    EXPECT_NEAR(deriv_weights[1], 0.8116903027548, tol);
    EXPECT_NEAR(deriv_weights[2], 0.0941548486226, tol);

    // Test point at 0. (on fourth QP)
    LagrangePolynomialDerivWeights(0., xs, deriv_weights);
    ASSERT_EQ(deriv_weights.size(), 3);
    EXPECT_NEAR(deriv_weights[0], -0.5, tol);
    EXPECT_NEAR(deriv_weights[1], 0., tol);
    EXPECT_NEAR(deriv_weights[2], 0.5, tol);

    // Test point at 0.4058451513773972 (on fifth QP)
    LagrangePolynomialDerivWeights(0.4058451513773972, xs, deriv_weights);
    ASSERT_EQ(deriv_weights.size(), 3);
    EXPECT_NEAR(deriv_weights[0], -0.0941548486226, tol);
    EXPECT_NEAR(deriv_weights[1], -0.8116903027548, tol);
    EXPECT_NEAR(deriv_weights[2], 0.9058451513774, tol);

    // Test point at 0.7415311855993945 (on sixth QP)
    LagrangePolynomialDerivWeights(0.7415311855993945, xs, deriv_weights);
    ASSERT_EQ(deriv_weights.size(), 3);
    EXPECT_NEAR(deriv_weights[0], 0.2415311855994, tol);
    EXPECT_NEAR(deriv_weights[1], -1.483062371199, tol);
    EXPECT_NEAR(deriv_weights[2], 1.241531185599, tol);

    // Test point at 0.9491079123427585 (on seventh QP)
    LagrangePolynomialDerivWeights(0.9491079123427585, xs, deriv_weights);
    ASSERT_EQ(deriv_weights.size(), 3);
    EXPECT_NEAR(deriv_weights[0], 0.4491079123428, tol);
    EXPECT_NEAR(deriv_weights[1], -1.898215824686, tol);
    EXPECT_NEAR(deriv_weights[2], 1.449107912343, tol);
}

TEST(InterpolationTest, LagrangePolynomialDerivWeights_FourthOrder) {
    // 5 nodes at GLL points for fourth order polynomial
    const std::vector<double> xs = {-1., -0.6546536707079771, 0., 0.6546536707079771, 1.};
    std::vector<double> weights;

    // Test point at x = -1 (on first node)
    LagrangePolynomialDerivWeights(-1., xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], -5., tol);
    EXPECT_NEAR(weights[1], 6.75650248872424, tol);
    EXPECT_NEAR(weights[2], -2.666666666666667, tol);
    EXPECT_NEAR(weights[3], 1.410164177942427, tol);
    EXPECT_NEAR(weights[4], -0.5, tol);

    // Test point at x = -0.6546536707079771 (on second node)
    LagrangePolynomialDerivWeights(-0.6546536707079771, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], -1.240990253030983, tol);
    EXPECT_NEAR(weights[1], 0., tol);
    EXPECT_NEAR(weights[2], 1.74574312188794, tol);
    EXPECT_NEAR(weights[3], -0.7637626158259736, tol);
    EXPECT_NEAR(weights[4], 0.2590097469690172, tol);

    // Test point at x = 0 (on third node)
    LagrangePolynomialDerivWeights(0., xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0.375, tol);
    EXPECT_NEAR(weights[1], -1.336584577695454, tol);
    EXPECT_NEAR(weights[2], 0., tol);
    EXPECT_NEAR(weights[3], 1.336584577695454, tol);
    EXPECT_NEAR(weights[4], -0.375, tol);

    // Test point at x = 0.6546536707079771 (on fourth node)
    LagrangePolynomialDerivWeights(0.6546536707079771, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], -0.2590097469690172, tol);
    EXPECT_NEAR(weights[1], 0.7637626158259736, tol);
    EXPECT_NEAR(weights[2], -1.74574312188794, tol);
    EXPECT_NEAR(weights[3], 0., tol);
    EXPECT_NEAR(weights[4], 1.240990253030983, tol);

    // Test point at x = 1 (on last node)
    LagrangePolynomialDerivWeights(1., xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0.5, tol);
    EXPECT_NEAR(weights[1], -1.410164177942427, tol);
    EXPECT_NEAR(weights[2], 2.666666666666667, tol);
    EXPECT_NEAR(weights[3], -6.756502488724241, tol);
    EXPECT_NEAR(weights[4], 5., tol);

    // Test point at x = -0.8 (between nodes 1 & 2)
    LagrangePolynomialDerivWeights(-0.8, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], -2.497, tol);
    EXPECT_NEAR(weights[1], 2.144324478146484, tol);
    EXPECT_NEAR(weights[2], 0.5546666666666656, tol);
    EXPECT_NEAR(weights[3], -0.31499114481315, tol);
    EXPECT_NEAR(weights[4], 0.1129999999999999, tol);

    // Test point at x = 0.1 (between nodes 3 & 4)
    LagrangePolynomialDerivWeights(0.1, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], 0.27725, tol);
    EXPECT_NEAR(weights[1], -0.896320373697923, tol);
    EXPECT_NEAR(weights[2], -0.6573333333333338, tol);
    EXPECT_NEAR(weights[3], 1.696653707031257, tol);
    EXPECT_NEAR(weights[4], -0.42025, tol);

    // Test point at x = 0.4 (between nodes 3 & 4)
    LagrangePolynomialDerivWeights(0.4, xs, weights);
    ASSERT_EQ(weights.size(), 5);
    EXPECT_NEAR(weights[0], -0.1210000000000001, tol);
    EXPECT_NEAR(weights[1], 0.4156426862650312, tol);
    EXPECT_NEAR(weights[2], -2.069333333333333, tol);
    EXPECT_NEAR(weights[3], 1.805690647068303, tol);
    EXPECT_NEAR(weights[4], -0.03099999999999978, tol);
}

TEST(InterpolationTest, LegendrePolynomial_ZerothOrder) {
    EXPECT_NEAR(LegendrePolynomial(0, -1.), 1., tol);
    EXPECT_NEAR(LegendrePolynomial(0, 0.), 1., tol);
    EXPECT_NEAR(LegendrePolynomial(0, 1.), 1., tol);
}

TEST(InterpolationTest, LegendrePolynomial_FirstOrder) {
    EXPECT_NEAR(LegendrePolynomial(1, -1.), -1., tol);
    EXPECT_NEAR(LegendrePolynomial(1, 0.), 0., tol);
    EXPECT_NEAR(LegendrePolynomial(1, 1.), 1., tol);
}

TEST(InterpolationTest, LegendrePolynomial_SecondOrder) {
    EXPECT_NEAR(LegendrePolynomial(2, -1.), 1., tol);
    EXPECT_NEAR(LegendrePolynomial(2, 0.), -0.5, tol);
    EXPECT_NEAR(LegendrePolynomial(2, 1.), 1., tol);
}

TEST(InterpolationTest, LegendrePolynomial_ThirdOrder) {
    EXPECT_NEAR(LegendrePolynomial(3, -1.), -1., tol);
    EXPECT_NEAR(LegendrePolynomial(3, 0.), 0., tol);
    EXPECT_NEAR(LegendrePolynomial(3, 1.), 1., tol);
}

TEST(InterpolationTest, LegendrePolynomial_FourthOrder) {
    EXPECT_NEAR(LegendrePolynomial(4, -1.), 1., tol);
    EXPECT_NEAR(LegendrePolynomial(4, -0.6546536707079771), -0.4285714285714286, tol);
    EXPECT_NEAR(LegendrePolynomial(4, 0.), 0.375, tol);
    EXPECT_NEAR(LegendrePolynomial(4, 0.6546536707079771), -0.4285714285714286, tol);
    EXPECT_NEAR(LegendrePolynomial(4, 1.), 1., tol);
}

TEST(InterpolationTest, GenerateGLLPoints_FirstOrderElement) {
    auto gll_points = GenerateGLLPoints(1);
    std::vector<double> expected = {-1., 1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

TEST(InterpolationTest, GenerateGLLPoints_SecondOrderElement) {
    auto gll_points = GenerateGLLPoints(2);
    std::vector<double> expected = {-1., 0., 1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

TEST(InterpolationTest, GenerateGLLPoints_FourthOrderElement) {
    auto gll_points = GenerateGLLPoints(4);
    std::vector<double> expected = {-1., -0.6546536707079771437983, 0., 0.654653670707977143798, 1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

TEST(InterpolationTest, GenerateGLLPoints_SixthOrderElement) {
    auto gll_points = GenerateGLLPoints(6);
    std::vector<double> expected = {-1., -0.8302238962785669, -0.46884879347071423,
                                    0.,  0.46884879347071423, 0.8302238962785669,
                                    1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

}  // namespace openturbine::tests
