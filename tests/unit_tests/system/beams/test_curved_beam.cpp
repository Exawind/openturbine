#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "elements/beams/beams.hpp"
#include "elements/beams/interpolation.hpp"
#include "system/beams/integrate_residual_vector.hpp"
#include "test_calculate.hpp"
#include "test_integrate_matrix.hpp"

/**
 * @file test_curved_beam.cpp
 * @brief Unit tests for S_t matrix, Residual, and/or o,p,q etc matrices of a beam element
 *
 * This file contains a series of unit tests that validate the curved beam consisting of
 * 3 nodes at GLL points for second order polynomial. The tests verify various aspects
 * of system matrices construction for a Beam element such as
 * - Lagrange polynomial/shape function and derivative weights i.e. LagrangePolynomialInterpWeights
 *   and LagrangePolynomialDerivWeights
 * - Jacobian calculation i.e. CalculateJacobian
 * - Residul vector integrtion i.e. beams::IntegrateResidualVectorElement
 * - Stiffness matrix integration i.e. beams::IntegrateStiffnessMatrixElement
 * - Inertia matrix integration i.e. beams::IntegrateInertiaMatrixElement
 *
 * Ref: https://github.com/michaelasprague/OpenTurbineTheory/tree/main/mathematica
 */

namespace openturbine::tests {

constexpr auto tol = 1.e-12;

TEST(CurvedBeamTests, LagrangePolynomialInterpWeight_SecondOrder_AtSpecifiedQuadraturePoints) {
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

TEST(CurvedBeamTests, LagrangePolynomialDerivWeight_SecondOrder_AtSpecifiedQuadraturePoints) {
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

TEST(CurvedBeamTests, CurvedBeamWithThreeNodes) {
    constexpr auto number_of_nodes = size_t{3U};  // element order = 2
    constexpr auto number_of_qps = size_t{7U};    // high order quadrature = 7

    const auto qp_weights = get_qp_weights<number_of_qps>({
        0.1294849661688697,  // weight at QP 1
        0.2797053914892766,  // weight at QP 2
        0.3818300505051189,  // weight at QP 3
        0.4179591836734694,  // weight at QP 4
        0.3818300505051189,  // weight at QP 5
        0.2797053914892766,  // weight at QP 6
        0.1294849661688697   // weight at QP 7
    });
    const auto qp_jacobian = get_qp_jacobian<number_of_qps>({
        2.631125640242,  // jacobian at QP 1
        2.54766419719,   // jacobian at QP 2
        2.501783068048,  // jacobian at QP 3
        2.598076211353,  // jacobian at QP 4
        2.843452426325,  // jacobian at QP 5
        3.134881687854,  // jacobian at QP 6
        3.34571483248    // jacobian at QP 7
    });
    const auto shape_interp = get_shape_interp<number_of_nodes, number_of_qps>({
        0.924956870807,    // h at qp 1, node 1
        0.0991941707284,   // h at qp 1, node 2
        -0.0241510415356,  // h at qp 1, node 3
        0.645699842408,    // h at qp 2, node 1
        0.450131500784,    // h at qp 2, node 2
        -0.0958313431915,  // h at qp 2, node 3
        0.285277719137,    // h at qp 3, node 1
        0.835289713103,    // h at qp 3, node 2
        -0.12056743224,    // h at qp 3, node 3
        0.,                // h at qp 4, node 1
        1.,                // h at qp 4, node 2
        0.,                // h at qp 4, node 3
        -0.12056743224,    // h at qp 5, node 1
        0.835289713103,    // h at qp 5, node 2
        0.285277719137,    // h at qp 5, node 3
        -0.0958313431915,  // h at qp 6, node 1
        0.450131500784,    // h at qp 6, node 2
        0.645699842408,    // h at qp 6, node 3
        -0.0241510415356,  // h at qp 7, node 1
        0.0991941707284,   // h at qp 7, node 2
        0.924956870807     // h at qp 7, node 3
    });

    const auto shape_deriv = get_shape_interp_deriv<number_of_nodes, number_of_qps>({
        -1.449107912343,   // dh at qp 1, node 1
        1.898215824686,    // dh at qp 1, node 2
        -0.4491079123428,  // dh at qp 1, node 3
        -1.241531185599,   // dh at qp 2, node 1
        1.483062371199,    // dh at qp 2, node 2
        -0.2415311855994,  // dh at qp 2, node 3
        -0.9058451513774,  // dh at qp 3, node 1
        0.8116903027548,   // dh at qp 3, node 2
        0.0941548486226,   // dh at qp 3, node 3
        -0.5,              // dh at qp 4, node 1
        0.,                // dh at qp 4, node 2
        0.5,               // dh at qp 4, node 3
        -0.0941548486226,  // dh at qp 5, node 1
        -0.8116903027548,  // dh at qp 5, node 2
        0.9058451513774,   // dh at qp 5, node 3
        0.2415311855994,   // dh at qp 6, node 1
        -1.483062371199,   // dh at qp 6, node 2
        1.241531185599,    // dh at qp 6, node 3
        0.4491079123428,   // dh at qp 7, node 1
        -1.898215824686,   // dh at qp 7, node 2
        1.449107912343     // dh at qp 7, node 3
    });

    using NodeVectorView = Kokkos::View<double[number_of_nodes][6]>;
    using QpVectorView = Kokkos::View<double[number_of_qps][6]>;

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = get_qp_Fc<number_of_qps>(
        {19377.66142402, -9011.48579619, 3582.628416357, 5433.695299839, 7030.727457672,
         -854.6894329742}
    );
    const auto qp_Fd =
        get_qp_Fd<number_of_qps>({0., 0., 0., -413.0521579912, 176.1330689806, 2677.142143344});
    const auto qp_Fi = get_qp_Fi<number_of_qps>(
        {0.02197622144767, -0.03476996186535, 0.005820529971857, -0.04149911138042,
         -0.07557306419557, -0.1562386708521}
    );

    const auto residual_vector_terms =
        Kokkos::View<double[1][number_of_nodes][6]>("residual_vector_terms");
}

}  // namespace openturbine::tests
