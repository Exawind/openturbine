#include "test_curved_beam.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "elements/beams/beams.hpp"
#include "elements/beams/calculate_jacobian.hpp"
#include "elements/beams/interpolation.hpp"
#include "system/beams/calculate_Ouu.hpp"
#include "system/beams/calculate_Puu.hpp"
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
 * - Elastic forces calculation -> CalculateForceFc
 * - Damping forces calculation -> CalculateForceFd
 * - Residul vector integrtion -> beams::IntegrateResidualVectorElement
 * - Stiffness matrix integration -> beams::IntegrateStiffnessMatrixElement
 * - Inertia matrix integration -> beams::IntegrateInertiaMatrixElement
 *
 * Ref: https://github.com/michaelasprague/OpenTurbineTheory/tree/main/mathematica
 */

namespace openturbine::tests {

using NodeVectorView = Kokkos::View<double[kNumNodes][6]>;  // 3 nodes, 6 components each
using QpVectorView = Kokkos::View<double[kNumQPs][6]>;      // 7 QPs, 6 components each

TEST(CurvedBeamTests, LagrangePolynomialInterpWeight_SecondOrder_AtSpecifiedQPs) {
    std::vector<double> weights;
    // Test interpolation weights at each QP against expected data from Mathematica script
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        LagrangePolynomialInterpWeights(kGaussQuadraturePoints[qp], kGLLNodes, weights);
        ASSERT_EQ(weights.size(), kNumNodes);
        EXPECT_NEAR(weights[0], kExpectedInterpWeights[qp][0], kDefaultTolerance);
        EXPECT_NEAR(weights[1], kExpectedInterpWeights[qp][1], kDefaultTolerance);
        EXPECT_NEAR(weights[2], kExpectedInterpWeights[qp][2], kDefaultTolerance);
    }
}

TEST(CurvedBeamTests, LagrangePolynomialDerivWeight_SecondOrder_AtSpecifiedQPs) {
    std::vector<double> deriv_weights;
    // Test derivative weights at each QP against expected data from Mathematica script
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        LagrangePolynomialDerivWeights(kGaussQuadraturePoints[qp], kGLLNodes, deriv_weights);
        ASSERT_EQ(deriv_weights.size(), kNumNodes);
        EXPECT_NEAR(deriv_weights[0], kExpectedDerivWeights[qp][0], kDefaultTolerance);
        EXPECT_NEAR(deriv_weights[1], kExpectedDerivWeights[qp][1], kDefaultTolerance);
        EXPECT_NEAR(deriv_weights[2], kExpectedDerivWeights[qp][2], kDefaultTolerance);
    }
}

TEST(CurvedBeamTests, CalculateJacobianForCurvedBeam) {
    const auto kNumNodes_per_elem = Kokkos::View<size_t*>("number_of_nodes_per_element", kNumElems);
    const auto kNumQPs_per_elem =
        Kokkos::View<size_t*>("number_of_quadrature_points_per_element", kNumElems);
    const auto shape_derivative =
        Kokkos::View<double***>("shape_derivative", kNumElems, kNumNodes, kNumQPs);
    const auto node_position_rotation =
        Kokkos::View<double** [7]>("node_position_rotation", kNumElems, kNumNodes);
    const auto qp_position_derivative =
        Kokkos::View<double** [3]>("position_derivative", kNumElems, kNumQPs);
    const auto qp_jacobian = Kokkos::View<double**>("jacobian", kNumElems, kNumQPs);

    const auto kNumNodes_host = Kokkos::create_mirror_view(kNumNodes_per_elem);
    const auto kNumQPs_host = Kokkos::create_mirror_view(kNumQPs_per_elem);
    const auto shape_derivative_host = Kokkos::create_mirror_view(shape_derivative);
    const auto node_position_rotation_host = Kokkos::create_mirror_view(node_position_rotation);
    const auto qp_position_derivative_host = Kokkos::create_mirror_view(qp_position_derivative);
    const auto qp_jacobian_host = Kokkos::create_mirror_view(qp_jacobian);

    // Set node positions
    for (size_t node = 0; node < kNumNodes; ++node) {
        node_position_rotation_host(0, node, 0) = kCurvedBeamNodes[node][0];
        node_position_rotation_host(0, node, 1) = kCurvedBeamNodes[node][1];
        node_position_rotation_host(0, node, 2) = kCurvedBeamNodes[node][2];
    }

    // Calculate shape function derivatives at each quadrature point
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        std::vector<double> weights{};
        LagrangePolynomialDerivWeights(kGaussQuadraturePoints[qp], kGLLNodes, weights);
        for (size_t node = 0; node < kNumNodes; ++node) {
            shape_derivative_host(0, node, qp) = weights[node];
        }
    }

    // Set number of nodes and quadrature points
    kNumNodes_host(0) = kNumNodes;
    kNumQPs_host(0) = kNumQPs;

    Kokkos::deep_copy(kNumNodes_per_elem, kNumNodes_host);
    Kokkos::deep_copy(kNumQPs_per_elem, kNumQPs_host);
    Kokkos::deep_copy(shape_derivative, shape_derivative_host);
    Kokkos::deep_copy(node_position_rotation, node_position_rotation_host);

    const auto calculate_jacobian =
        CalculateJacobian{kNumNodes_per_elem,     kNumQPs_per_elem,       shape_derivative,
                          node_position_rotation, qp_position_derivative, qp_jacobian};
    Kokkos::parallel_for("calculate_jacobian", 1, calculate_jacobian);
    Kokkos::deep_copy(qp_jacobian_host, qp_jacobian);

    // Validate the calculated jacobians against expected values from Mathematica script
    ASSERT_EQ(qp_jacobian_host.extent(0), kNumElems);  // 1 element
    ASSERT_EQ(qp_jacobian_host.extent(1), kNumQPs);    // 7 quadrature points

    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        EXPECT_NEAR(qp_jacobian_host(0, qp), kExpectedJacobians[qp], kDefaultTolerance)
            << "Jacobian mismatch at quadrature point " << qp;
    }

    // Verify that position derivatives are unit vectors
    ASSERT_EQ(qp_position_derivative_host.extent(0), kNumElems);  // 1 element
    ASSERT_EQ(qp_position_derivative_host.extent(1), kNumQPs);    // 7 quadrature points
    ASSERT_EQ(qp_position_derivative_host.extent(2), 3);          // 3 dimensions

    Kokkos::deep_copy(qp_position_derivative, qp_position_derivative_host);
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        const auto magnitude = std::sqrt(
            qp_position_derivative_host(0, qp, 0) * qp_position_derivative_host(0, qp, 0) +
            qp_position_derivative_host(0, qp, 1) * qp_position_derivative_host(0, qp, 1) +
            qp_position_derivative_host(0, qp, 2) * qp_position_derivative_host(0, qp, 2)
        );
        EXPECT_NEAR(magnitude, 1., kDefaultTolerance);  // unit vector
    }
}

TEST(CurvedBeamTests, CalculateForceFcForCurvedBeam) {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto strain = Kokkos::View<double[6]>("strain");
    const auto Fc = Kokkos::View<double[6]>("Fc");

    const auto Cuu_host = Kokkos::create_mirror_view(Cuu);
    const auto strain_host = Kokkos::create_mirror_view(strain);
    const auto Fc_host = Kokkos::create_mirror_view(Fc);

    // Set stiffness matrix in global frame i.e. Cuu matrix
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

TEST(CurvedBeamTests, IntegrateResidualVectorForCurvedBeam) {
    const auto qp_weights = get_qp_weights<kNumQPs>(kGaussQuadratureWeights);
    const auto qp_jacobian = get_qp_jacobian<kNumQPs>(kExpectedJacobians);
    const auto shape_interp = get_shape_interp<kNumNodes, kNumQPs>(kInterpWeightsFlat);
    const auto shape_deriv = get_shape_interp_deriv<kNumNodes, kNumQPs>(kDerivWeightsFlat);

    const auto node_FX = NodeVectorView("node_FX");
    const auto qp_Fc = get_qp_Fc<kNumQPs>(kFc);
    const auto qp_Fd = get_qp_Fd<kNumQPs>(kFd);
    const auto qp_Fi = get_qp_Fi<kNumQPs>(kFi);
    const auto qp_Fe = QpVectorView("qp_Fe");
    const auto qp_Fg = QpVectorView("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[kNumElems][kNumNodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", kNumNodes,
        beams::IntegrateResidualVectorElement{
            0U, kNumQPs, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc, qp_Fd,
            qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    const auto resid_exact = Kokkos::View<const double[kNumElems][kNumNodes][6], Kokkos::HostSpace>(
        kExpectedResidualVector.data()
    );

    const auto residual_vector_terms_mirror = Kokkos::create_mirror(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, resid_exact, 1e-8);
}

TEST(CurvedBeamTests, CalculateOuuMatrixForCurvedBeam) {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    const auto M_tilde = Kokkos::View<double[3][3]>("M_tilde");
    const auto N_tilde = Kokkos::View<double[3][3]>("N_tilde");
    const auto Ouu = Kokkos::View<double[6][6]>("Ouu");

    const auto Cuu_host = Kokkos::create_mirror_view(Cuu);
    const auto x0pupSS_host = Kokkos::create_mirror_view(x0pupSS);
    const auto M_tilde_host = Kokkos::create_mirror_view(M_tilde);
    const auto N_tilde_host = Kokkos::create_mirror_view(N_tilde);

    // Set stiffness matrix in global frame i.e. Cuu matrix
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            Cuu_host(i, j) = kCurvedBeamCuu[i][j];
        }
    }

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

    // Set M_tilde values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            M_tilde_host(i, j) = kCurvedBeamM_tilde[i][j];
        }
    }

    // Set N_tilde values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            N_tilde_host(i, j) = kCurvedBeamN_tilde[i][j];
        }
    }

    Kokkos::deep_copy(Cuu, Cuu_host);
    Kokkos::deep_copy(x0pupSS, x0pupSS_host);
    Kokkos::deep_copy(M_tilde, M_tilde_host);

    Kokkos::parallel_for(
        "CalculateOuu", 1,
        KOKKOS_LAMBDA(size_t) { beams::CalculateOuu(Cuu, x0pupSS, M_tilde, N_tilde, Ouu); }
    );

    const auto Ouu_host = Kokkos::create_mirror_view(Ouu);
    Kokkos::deep_copy(Ouu_host, Ouu);

    // Validate against expected values from Mathematica script
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(Ouu_host(i, j), kExpectedOuu[i][j], 1e-6)
                << "Ouu mismatch at component (" << i << ", " << j << ")";
        }
    }
}

TEST(CurvedBeamTests, CalculatePuuMatrixForCurvedBeam) {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    const auto N_tilde = Kokkos::View<double[3][3]>("N_tilde");
    const auto Puu = Kokkos::View<double[6][6]>("Puu");

    const auto Cuu_host = Kokkos::create_mirror_view(Cuu);
    const auto x0pupSS_host = Kokkos::create_mirror_view(x0pupSS);
    const auto N_tilde_host = Kokkos::create_mirror_view(N_tilde);

    // Set stiffness matrix in global frame i.e. Cuu matrix
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            Cuu_host(i, j) = kCurvedBeamCuu[i][j];
        }
    }

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

    // Set N_tilde values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            N_tilde_host(i, j) = kCurvedBeamN_tilde[i][j];
        }
    }

    Kokkos::deep_copy(Cuu, Cuu_host);
    Kokkos::deep_copy(x0pupSS, x0pupSS_host);
    Kokkos::deep_copy(N_tilde, N_tilde_host);

    Kokkos::parallel_for(
        "CalculatePuu", 1, KOKKOS_LAMBDA(size_t) { beams::CalculatePuu(Cuu, x0pupSS, N_tilde, Puu); }
    );

    const auto Puu_host = Kokkos::create_mirror_view(Puu);
    Kokkos::deep_copy(Puu_host, Puu);

    // Validate against expected values from Mathematica script
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(Puu_host(i, j), kExpectedPuu[i][j], 1e-6)
                << "Puu mismatch at component (" << i << ", " << j << ")";
        }
    }
}

}  // namespace openturbine::tests
