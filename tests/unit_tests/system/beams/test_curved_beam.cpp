#include "test_curved_beam.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "elements/beams/beams.hpp"
#include "elements/beams/calculate_jacobian.hpp"
#include "elements/beams/interpolation.hpp"
#include "system/beams/calculate_force_FC.hpp"
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

TEST(CurvedBeamTests, LagrangePolynomialInterpWeight_SecondOrder_AtSpecifiedQuadraturePoints) {
    std::vector<double> weights;

    // Test interpolation weights at each quadrature point
    for (size_t qp = 0; qp < kGaussQuadraturePoints.size(); ++qp) {
        LagrangePolynomialInterpWeights(kGaussQuadraturePoints[qp], kGLLNodes, weights);
        ASSERT_EQ(weights.size(), 3);
        EXPECT_NEAR(weights[0], kExpectedInterpWeights[qp][0], kTolerance);
        EXPECT_NEAR(weights[1], kExpectedInterpWeights[qp][1], kTolerance);
        EXPECT_NEAR(weights[2], kExpectedInterpWeights[qp][2], kTolerance);
    }
}

TEST(CurvedBeamTests, LagrangePolynomialDerivWeight_SecondOrder_AtSpecifiedQuadraturePoints) {
    std::vector<double> deriv_weights;

    // Test derivative weights at each quadrature point
    for (size_t qp = 0; qp < kGaussQuadraturePoints.size(); ++qp) {
        LagrangePolynomialDerivWeights(kGaussQuadraturePoints[qp], kGLLNodes, deriv_weights);
        ASSERT_EQ(deriv_weights.size(), 3);
        EXPECT_NEAR(deriv_weights[0], kExpectedDerivWeights[qp][0], kTolerance);
        EXPECT_NEAR(deriv_weights[1], kExpectedDerivWeights[qp][1], kTolerance);
        EXPECT_NEAR(deriv_weights[2], kExpectedDerivWeights[qp][2], kTolerance);
    }
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

    // Set node positions
    for (size_t node = 0; node < num_nodes; ++node) {
        host_node_position_rotation(0, node, 0) = kCurvedBeamNodes[node][0];
        host_node_position_rotation(0, node, 1) = kCurvedBeamNodes[node][1];
        host_node_position_rotation(0, node, 2) = kCurvedBeamNodes[node][2];
    }

    // Calculate shape function derivatives at each quadrature point
    for (size_t qp = 0; qp < num_qps; ++qp) {
        std::vector<double> weights{};
        LagrangePolynomialDerivWeights(kGaussQuadraturePoints[qp], kGLLNodes, weights);
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

    // Validate the calculated jacobians against expected values from Mathematica script
    ASSERT_EQ(host_qp_jacobian.extent(0), num_elems);  // 1 element
    ASSERT_EQ(host_qp_jacobian.extent(1), num_qps);    // 7 quadrature points

    for (size_t qp = 0; qp < num_qps; ++qp) {
        EXPECT_NEAR(host_qp_jacobian(0, qp), kExpectedJacobians[qp], kTolerance)
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

TEST(CurvedBeamTests, CalculateForceFcForCurvedBeam) {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto strain = Kokkos::View<double[6]>("strain");
    const auto Fc = Kokkos::View<double[6]>("Fc");

    const auto Cuu_host = Kokkos::create_mirror_view(Cuu);
    const auto strain_host = Kokkos::create_mirror_view(strain);
    const auto Fc_host = Kokkos::create_mirror_view(Fc);

    // Stiffness matrix
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            Cuu_host(i, j) = kCurvedBeamCuu[i][j];
        }
    }
    // Strain
    for (size_t i = 0; i < 6; ++i) {
        strain_host(i) = kCurvedBeamStrain[i];
    }

    Kokkos::deep_copy(Cuu, Cuu_host);
    Kokkos::deep_copy(strain, strain_host);

    Kokkos::parallel_for(
        "CalculateForceFc", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateForceFC(Cuu, strain, Fc); }
    );
    Kokkos::deep_copy(Fc_host, Fc);

    // Validate against expected values from Mathematica script
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(Fc_host(i), kExpectedFc[i], 1e-6) << "Force Fc mismatch at component " << i;
    }
}

}  // namespace openturbine::tests
