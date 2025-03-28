#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "elements/beams/beams.hpp"
#include "elements/beams/calculate_jacobian.hpp"
#include "elements/beams/interpolation.hpp"
#include "system/beams/integrate_residual_vector.hpp"
#include "test_calculate.hpp"
#include "test_integrate_matrix.hpp"

/**
 * @file test_curved_beam.cpp
 * @brief Unit tests for S_t matrix, Residual, O, P, Q etc. matrices of a beam element
 *
 * This file contains a series of unit tests that validate a curved beam problem consisting of
 * 3 nodes at GLL points (for second order polynomial). The validation dataset was created in
 * Mathematica during openturbine theory development.
 *
 * The Mathematica notebooks used to generate the validation dataset are:
 * - Run1-RotationLibrary.nb
 * - Run2-ShapeFunctions.nb
 * - Run3-ReferenceLineDefinition.nb
 * - Run4-TestInterpolatedValues.nb
 * - Run5-DynamicIterationMatrix.nb
 *
 * The tests verify various aspects of system matrices construction for a Beam element such as:
 * - Lagrange polynomial/shape function -> LagrangePolynomialInterpWeights
 *   and derivative weights -> LagrangePolynomialDerivWeights
 * - Jacobian calculation -> CalculateJacobian
 * - Residul vector integrtion -> beams::IntegrateResidualVectorElement
 * - Stiffness matrix integration -> beams::IntegrateStiffnessMatrixElement
 * - Inertia matrix integration -> beams::IntegrateInertiaMatrixElement
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
    // -0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
    // 0.7415311855993945, 0.9491079123427585

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
    // -0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
    // 0.7415311855993945, 0.9491079123427585

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

TEST(CurvedBeamTests, CalculateJacobianForCurvedBeam) {
    constexpr size_t num_elems{1};
    constexpr size_t num_nodes{3};
    constexpr size_t num_qps{7};

    const auto num_nodes_per_elem = Kokkos::View<size_t*>("num_nodes", num_elems);
    const auto num_qps_per_elem = Kokkos::View<size_t*>("num_qps", num_elems);
    const auto shape_derivative =
        Kokkos::View<double***>("shape_derivative", num_elems, num_nodes, num_qps);
    const auto node_position_rotation =
        Kokkos::View<double** [7]>("node_position_rotation", num_elems, num_nodes);
    const auto qp_position_derivative =
        Kokkos::View<double** [3]>("position_derivative", num_elems, num_qps);
    const auto qp_jacobian = Kokkos::View<double**>("jacobian", num_elems, num_qps);

    const auto host_num_nodes = Kokkos::create_mirror_view(num_nodes_per_elem);
    const auto host_num_qps = Kokkos::create_mirror_view(num_qps_per_elem);
    const auto host_shape_derivative = Kokkos::create_mirror_view(shape_derivative);
    const auto host_node_position_rotation = Kokkos::create_mirror_view(node_position_rotation);
    const auto host_qp_position_derivative = Kokkos::create_mirror_view(qp_position_derivative);
    const auto host_qp_jacobian = Kokkos::create_mirror_view(qp_jacobian);

    // Node positions at GLL points (from Mathematica script)
    // Node 1
    host_node_position_rotation(0, 0, 0) = 0.;  // x
    host_node_position_rotation(0, 0, 1) = 0.;  // y
    host_node_position_rotation(0, 0, 2) = 0.;  // z
    // Node 2
    host_node_position_rotation(0, 1, 0) = 2.5;     // x
    host_node_position_rotation(0, 1, 1) = -0.125;  // y
    host_node_position_rotation(0, 1, 2) = 0.;      // z
    // Node 3
    host_node_position_rotation(0, 2, 0) = 5.;   // x
    host_node_position_rotation(0, 2, 1) = 1.;   // y
    host_node_position_rotation(0, 2, 2) = -1.;  // z

    // Define the quadrature points for 7-point Gauss quadrature
    constexpr std::array<double, 7> qp_locations = {-0.9491079123427585, -0.7415311855993945,
                                                    -0.4058451513773972, 0.,
                                                    0.4058451513773972,  0.7415311855993945,
                                                    0.9491079123427585};

    // Calculate shape function derivatives at each quadrature point
    const std::vector<double> nodes = {-1., 0., 1.};
    for (size_t qp = 0; qp < num_qps; ++qp) {
        std::vector<double> weights{};
        LagrangePolynomialDerivWeights(qp_locations[qp], nodes, weights);
        for (size_t node = 0; node < num_nodes; ++node) {
            host_shape_derivative(0, node, qp) = weights[node];
        }
    }

    host_num_nodes(0) = num_nodes;
    host_num_qps(0) = num_qps;

    Kokkos::deep_copy(num_nodes_per_elem, host_num_nodes);
    Kokkos::deep_copy(num_qps_per_elem, host_num_qps);
    Kokkos::deep_copy(shape_derivative, host_shape_derivative);
    Kokkos::deep_copy(node_position_rotation, host_node_position_rotation);

    CalculateJacobian calculate_jacobian{num_nodes_per_elem,     num_qps_per_elem,
                                         shape_derivative,       node_position_rotation,
                                         qp_position_derivative, qp_jacobian};
    Kokkos::parallel_for("calculate_jacobian", 1, calculate_jacobian);
    Kokkos::deep_copy(host_qp_jacobian, qp_jacobian);

    // Expected jacobians at each quadrature point (from Mathematica script)
    constexpr std::array<double, 7> expected_jacobians = {
        2.631125640242,  // jacobian at QP 1
        2.54766419719,   // jacobian at QP 2
        2.501783068048,  // jacobian at QP 3
        2.598076211353,  // jacobian at QP 4
        2.843452426325,  // jacobian at QP 5
        3.134881687854,  // jacobian at QP 6
        3.34571483248    // jacobian at QP 7
    };

    // Validate the calculated jacobians against expected values
    ASSERT_EQ(host_qp_jacobian.extent(0), num_elems);  // 1 element
    ASSERT_EQ(host_qp_jacobian.extent(1), num_qps);    // 7 quadrature points

    const double tolerance = 1e-10;
    for (size_t qp = 0; qp < num_qps; ++qp) {
        EXPECT_NEAR(host_qp_jacobian(0, qp), expected_jacobians[qp], tolerance)
            << "Jacobian mismatch at quadrature point " << qp;
    }

    // Verify that position derivatives are unit vectors
    ASSERT_EQ(host_qp_position_derivative.extent(0), num_elems);  // 1 element
    ASSERT_EQ(host_qp_position_derivative.extent(1), num_qps);    // 7 quadrature points
    ASSERT_EQ(host_qp_position_derivative.extent(2), 3);          // 3 dimensions

    Kokkos::deep_copy(host_qp_position_derivative, qp_position_derivative);
    for (size_t qp = 0; qp < num_qps; ++qp) {
        const auto magnitude = std::sqrt(
            host_qp_position_derivative(0, qp, 0) * host_qp_position_derivative(0, qp, 0) +
            host_qp_position_derivative(0, qp, 1) * host_qp_position_derivative(0, qp, 1) +
            host_qp_position_derivative(0, qp, 2) * host_qp_position_derivative(0, qp, 2)
        );
        EXPECT_NEAR(magnitude, 1., 1e-12);  // unit vector
    }
}

}  // namespace openturbine::tests
