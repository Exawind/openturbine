#include "test_curved_beam.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "elements/beams/beams.hpp"
#include "elements/beams/calculate_jacobian.hpp"
#include "elements/beams/interpolation.hpp"
#include "system/beams/calculate_force_FC.hpp"
#include "system/beams/calculate_force_FD.hpp"
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

    // Test interpolation weights at each QP against expected data from Mathematica script
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

    // Test derivative weights at each QP against expected data from Mathematica script
    for (size_t qp = 0; qp < kGaussQuadraturePoints.size(); ++qp) {
        LagrangePolynomialDerivWeights(kGaussQuadraturePoints[qp], kGLLNodes, deriv_weights);
        ASSERT_EQ(deriv_weights.size(), 3);
        EXPECT_NEAR(deriv_weights[0], kExpectedDerivWeights[qp][0], kTolerance);
        EXPECT_NEAR(deriv_weights[1], kExpectedDerivWeights[qp][1], kTolerance);
        EXPECT_NEAR(deriv_weights[2], kExpectedDerivWeights[qp][2], kTolerance);
    }
}

TEST(CurvedBeamTests, CalculateJacobianForCurvedBeam) {
    const auto num_nodes_per_elem = Kokkos::View<size_t*>("num_nodes", num_elems);
    const auto num_qps_per_elem = Kokkos::View<size_t*>("num_qps", num_elems);
    const auto shape_derivative =
        Kokkos::View<double***>("shape_derivative", num_elems, num_nodes, num_qps);
    const auto node_position_rotation =
        Kokkos::View<double** [7]>("node_position_rotation", num_elems, num_nodes);
    const auto qp_position_derivative =
        Kokkos::View<double** [3]>("position_derivative", num_elems, num_qps);
    const auto qp_jacobian = Kokkos::View<double**>("jacobian", num_elems, num_qps);

    const auto num_nodes_host = Kokkos::create_mirror_view(num_nodes_per_elem);
    const auto num_qps_host = Kokkos::create_mirror_view(num_qps_per_elem);
    const auto shape_derivative_host = Kokkos::create_mirror_view(shape_derivative);
    const auto node_position_rotation_host = Kokkos::create_mirror_view(node_position_rotation);
    const auto qp_position_derivative_host = Kokkos::create_mirror_view(qp_position_derivative);
    const auto qp_jacobian_host = Kokkos::create_mirror_view(qp_jacobian);

    // Set node positions
    for (size_t node = 0; node < num_nodes; ++node) {
        node_position_rotation_host(0, node, 0) = kCurvedBeamNodes[node][0];
        node_position_rotation_host(0, node, 1) = kCurvedBeamNodes[node][1];
        node_position_rotation_host(0, node, 2) = kCurvedBeamNodes[node][2];
    }

    // Calculate shape function derivatives at each quadrature point
    for (size_t qp = 0; qp < num_qps; ++qp) {
        std::vector<double> weights{};
        LagrangePolynomialDerivWeights(kGaussQuadraturePoints[qp], kGLLNodes, weights);
        for (size_t node = 0; node < num_nodes; ++node) {
            shape_derivative_host(0, node, qp) = weights[node];
        }
    }

    // Set number of nodes and quadrature points
    num_nodes_host(0) = num_nodes;
    num_qps_host(0) = num_qps;

    Kokkos::deep_copy(num_nodes_per_elem, num_nodes_host);
    Kokkos::deep_copy(num_qps_per_elem, num_qps_host);
    Kokkos::deep_copy(shape_derivative, shape_derivative_host);
    Kokkos::deep_copy(node_position_rotation, node_position_rotation_host);

    CalculateJacobian calculate_jacobian{num_nodes_per_elem,     num_qps_per_elem,
                                         shape_derivative,       node_position_rotation,
                                         qp_position_derivative, qp_jacobian};
    Kokkos::parallel_for("calculate_jacobian", 1, calculate_jacobian);
    Kokkos::deep_copy(qp_jacobian_host, qp_jacobian);

    // Validate the calculated jacobians against expected values from Mathematica script
    ASSERT_EQ(qp_jacobian_host.extent(0), num_elems);  // 1 element
    ASSERT_EQ(qp_jacobian_host.extent(1), num_qps);    // 7 quadrature points

    for (size_t qp = 0; qp < num_qps; ++qp) {
        EXPECT_NEAR(qp_jacobian_host(0, qp), kExpectedJacobians[qp], kTolerance)
            << "Jacobian mismatch at quadrature point " << qp;
    }

    // Verify that position derivatives are unit vectors
    ASSERT_EQ(qp_position_derivative_host.extent(0), num_elems);  // 1 element
    ASSERT_EQ(qp_position_derivative_host.extent(1), num_qps);    // 7 quadrature points
    ASSERT_EQ(qp_position_derivative_host.extent(2), 3);          // 3 dimensions

    Kokkos::deep_copy(qp_position_derivative, qp_position_derivative_host);
    for (size_t qp = 0; qp < num_qps; ++qp) {
        const auto magnitude = std::sqrt(
            qp_position_derivative_host(0, qp, 0) * qp_position_derivative_host(0, qp, 0) +
            qp_position_derivative_host(0, qp, 1) * qp_position_derivative_host(0, qp, 1) +
            qp_position_derivative_host(0, qp, 2) * qp_position_derivative_host(0, qp, 2)
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

    // Set stiffness matrix i.e. Cuu matrix
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            Cuu_host(i, j) = kCurvedBeamCuu[i][j];
        }
    }
    // Set strain vector
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

TEST(CurvedBeamTests, CalculateForceFdForCurvedBeam) {
    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    const auto Fc = Kokkos::View<double[6]>("Fc");
    const auto Fd = Kokkos::View<double[6]>("Fd");

    const auto x0pupSS_host = Kokkos::create_mirror_view(x0pupSS);
    const auto Fc_host = Kokkos::create_mirror_view(Fc);
    const auto Fd_host = Kokkos::create_mirror_view(Fd);

    // Set x0pupSS values
    x0pupSS_host(0, 0) = 0.;
    x0pupSS_host(0, 1) = -kStrainInterpolationHolder[2];
    x0pupSS_host(0, 2) = kStrainInterpolationHolder[1];
    x0pupSS_host(1, 0) = kStrainInterpolationHolder[2];
    x0pupSS_host(1, 1) = 0.;
    x0pupSS_host(1, 2) = -kStrainInterpolationHolder[0];
    x0pupSS_host(2, 0) = -kStrainInterpolationHolder[1];
    x0pupSS_host(2, 1) = kStrainInterpolationHolder[0];
    x0pupSS_host(2, 2) = 0.;

    // Set Fc values
    for (size_t i = 0; i < 6; ++i) {
        Fc_host(i) = kExpectedFc[i];
    }

    Kokkos::deep_copy(x0pupSS, x0pupSS_host);
    Kokkos::deep_copy(Fc, Fc_host);

    Kokkos::parallel_for(
        "CalculateForceFd", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateForceFD(x0pupSS, Fc, Fd); }
    );
    Kokkos::deep_copy(Fd_host, Fd);

    // Validate results against expected values from Mathematica script
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(Fd_host(i), kExpectedFd[i], 1e-6) << "Force Fd mismatch at component " << i;
    }
}

}  // namespace openturbine::tests
