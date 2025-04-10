#include "test_curved_beam.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "elements/beams/beams.hpp"
#include "elements/beams/calculate_jacobian.hpp"
#include "elements/beams/interpolation.hpp"
#include "system/beams/calculate_Ouu.hpp"
#include "system/beams/calculate_Puu.hpp"
#include "system/beams/calculate_Quu.hpp"
#include "system/beams/calculate_force_FC.hpp"
#include "system/beams/calculate_force_FD.hpp"
#include "system/beams/integrate_residual_vector.hpp"
#include "system/beams/integrate_stiffness_matrix.hpp"
#include "system/beams/rotate_section_matrix.hpp"
#include "test_calculate.hpp"
#include "test_integrate_matrix.hpp"

/**
 * @file test_curved_beam.cpp
 * @brief Unit tests for S_t matrix, Residual, O, P, Q matrices, etc., of a beam element
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
 * - Residual vector integration -> beams::IntegrateResidualVectorElement
 * - Rotate sectional stiffness matrix -> beams::RotateSectionMatrix
 * - O matrix calculation -> CalculateOuu
 * - P matrix calculation -> CalculatePuu
 * - Q matrix calculation -> CalculateQuu
 * - Stiffness matrix integration -> beams::IntegrateStiffnessMatrixElement
 *
 * Ref: https://github.com/michaelasprague/OpenTurbineTheory/tree/main/mathematica
 */

namespace openturbine::tests::curved_beam {

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

    const auto kNumNodes_mirror = Kokkos::create_mirror_view(kNumNodes_per_elem);
    const auto kNumQPs_mirror = Kokkos::create_mirror_view(kNumQPs_per_elem);
    const auto shape_derivative_mirror = Kokkos::create_mirror_view(shape_derivative);
    const auto node_position_rotation_mirror = Kokkos::create_mirror_view(node_position_rotation);
    const auto qp_position_derivative_mirror = Kokkos::create_mirror_view(qp_position_derivative);
    const auto qp_jacobian_mirror = Kokkos::create_mirror_view(qp_jacobian);

    // Set node positions
    auto node_positions =
        Kokkos::subview(node_position_rotation_mirror, 0, Kokkos::ALL, Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(node_positions, kCurvedBeamNodes);

    // Calculate shape function derivatives at each quadrature point
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        std::vector<double> weights{};
        LagrangePolynomialDerivWeights(kGaussQuadraturePoints[qp], kGLLNodes, weights);
        for (size_t node = 0; node < kNumNodes; ++node) {
            shape_derivative_mirror(0, node, qp) = weights[node];
        }
    }

    // Set number of nodes and quadrature points
    kNumNodes_mirror(0) = kNumNodes;
    kNumQPs_mirror(0) = kNumQPs;

    Kokkos::deep_copy(kNumNodes_per_elem, kNumNodes_mirror);
    Kokkos::deep_copy(kNumQPs_per_elem, kNumQPs_mirror);
    Kokkos::deep_copy(shape_derivative, shape_derivative_mirror);
    Kokkos::deep_copy(node_position_rotation, node_position_rotation_mirror);

    const auto calculate_jacobian =
        CalculateJacobian{kNumNodes_per_elem,     kNumQPs_per_elem,       shape_derivative,
                          node_position_rotation, qp_position_derivative, qp_jacobian};
    Kokkos::parallel_for("calculate_jacobian", 1, calculate_jacobian);
    Kokkos::deep_copy(qp_jacobian_mirror, qp_jacobian);

    // Validate the calculated jacobians against expected values from Mathematica script
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        EXPECT_NEAR(qp_jacobian_mirror(0, qp), kExpectedJacobians[qp], kDefaultTolerance)
            << "Jacobian mismatch at quadrature point " << qp;
    }

    // Verify that position derivatives are unit vectors
    Kokkos::deep_copy(qp_position_derivative_mirror, qp_position_derivative);
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        const auto magnitude = std::sqrt(
            qp_position_derivative_mirror(0, qp, 0) * qp_position_derivative_mirror(0, qp, 0) +
            qp_position_derivative_mirror(0, qp, 1) * qp_position_derivative_mirror(0, qp, 1) +
            qp_position_derivative_mirror(0, qp, 2) * qp_position_derivative_mirror(0, qp, 2)
        );
        EXPECT_NEAR(magnitude, 1., kDefaultTolerance);  // unit vector
    }
}

void TestCalculateForceFc() {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto strain = Kokkos::View<double[6]>("strain");
    const auto Fc = Kokkos::View<double[6]>("Fc");

    // Set Cuu data
    const auto Cuu_mirror = Kokkos::create_mirror_view(Cuu);
    Kokkos::deep_copy(Cuu_mirror, kCurvedBeamCuu);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    // Set strain data
    const auto strain_mirror = Kokkos::create_mirror_view(strain);
    Kokkos::deep_copy(strain_mirror, kCurvedBeamStrain);
    Kokkos::deep_copy(strain, strain_mirror);

    Kokkos::parallel_for(
        "CalculateForceFc", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateForceFC(Cuu, strain, Fc); }
    );

    const auto Fc_exact = Kokkos::View<double[6]>("Fc");
    Kokkos::deep_copy(Fc_exact, kExpectedFc);
    const auto Fc_mirror = Kokkos::create_mirror_view(Fc);
    Kokkos::deep_copy(Fc_mirror, Fc);
    CompareWithExpected(Fc_mirror, Fc_exact, 1e-6);
}

TEST(CurvedBeamTests, CalculateForceFcForCurvedBeam) {
    TestCalculateForceFc();
}

void TestCalculateForceFd() {
    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    const auto Fc = Kokkos::View<double[6]>("Fc");
    const auto Fd = Kokkos::View<double[6]>("Fd");

    // Set x0pupSS data
    const auto x0pupSS_mirror = Kokkos::create_mirror_view(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, kX0pupSS);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    // Set Fc data
    const auto Fc_mirror = Kokkos::create_mirror_view(Fc);
    Kokkos::deep_copy(Fc_mirror, kExpectedFc);
    Kokkos::deep_copy(Fc, Fc_mirror);

    Kokkos::parallel_for(
        "CalculateForceFd", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateForceFD(x0pupSS, Fc, Fd); }
    );

    const auto Fd_exact = Kokkos::View<double[6]>("Fd");
    Kokkos::deep_copy(Fd_exact, kExpectedFd);
    const auto Fd_mirror = Kokkos::create_mirror_view(Fd);
    Kokkos::deep_copy(Fd_mirror, Fd);
    CompareWithExpected(Fd_mirror, Fd_exact, 1e-6);
}

TEST(CurvedBeamTests, CalculateForceFdForCurvedBeam) {
    TestCalculateForceFd();
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

    const auto residual_vector_exact =
        Kokkos::View<double[kNumElems][kNumNodes][6], Kokkos::HostSpace>::const_type(
            kExpectedResidualVector.data()
        );

    auto residual_vector_terms_mirror = Kokkos::create_mirror_view(residual_vector_terms);
    Kokkos::deep_copy(residual_vector_terms_mirror, residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, residual_vector_exact, 1e-8);
}

void TestRotateSectionMatrixForCurvedBeam() {
    const auto xr = Kokkos::View<double[4]>("xr");
    const auto Cstar = Kokkos::View<double[6][6]>("Cstar");
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");

    // Set quaternion data
    const auto xr_mirror = Kokkos::create_mirror_view(xr);
    Kokkos::deep_copy(xr_mirror, kCurvedBeamXr);
    Kokkos::deep_copy(xr, xr_mirror);

    // Set material stiffness matrix data
    const auto Cstar_mirror = Kokkos::create_mirror_view(Cstar);
    Kokkos::deep_copy(Cstar_mirror, kCurvedBeamCstar);
    Kokkos::deep_copy(Cstar, Cstar_mirror);

    Kokkos::parallel_for(
        "RotateSectionMatrix", 1,
        KOKKOS_LAMBDA(size_t) { beams::RotateSectionMatrix(xr, Cstar, Cuu); }
    );

    const auto Cuu_exact = Kokkos::View<double[6][6]>("Cuu");
    Kokkos::deep_copy(Cuu_exact, kExpectedCuu);
    const auto Cuu_mirror = Kokkos::create_mirror_view(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu);
    CompareWithExpected(Cuu_mirror, Cuu_exact, 1e-6);
}

TEST(CurvedBeamTests, CalculateRotatedStiffnessMatrixForCurvedBeam) {
    TestRotateSectionMatrixForCurvedBeam();
}

void TestCalculateOuu() {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    const auto M_tilde = Kokkos::View<double[3][3]>("M_tilde");
    const auto N_tilde = Kokkos::View<double[3][3]>("N_tilde");
    const auto Ouu = Kokkos::View<double[6][6]>("Ouu");

    // Set Cuu data
    const auto Cuu_mirror = Kokkos::create_mirror_view(Cuu);
    Kokkos::deep_copy(Cuu_mirror, kCurvedBeamCuu);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    // Set x0pupSS data
    const auto x0pupSS_mirror = Kokkos::create_mirror_view(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, kX0pupSS);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    // Set M_tilde data
    const auto M_tilde_mirror = Kokkos::create_mirror_view(M_tilde);
    Kokkos::deep_copy(M_tilde_mirror, kCurvedBeamM_tilde);
    Kokkos::deep_copy(M_tilde, M_tilde_mirror);

    // Set N_tilde data
    const auto N_tilde_mirror = Kokkos::create_mirror_view(N_tilde);
    Kokkos::deep_copy(N_tilde_mirror, kCurvedBeamN_tilde);
    Kokkos::deep_copy(N_tilde, N_tilde_mirror);

    Kokkos::parallel_for(
        "CalculateOuu", 1,
        KOKKOS_LAMBDA(size_t) { beams::CalculateOuu(Cuu, x0pupSS, M_tilde, N_tilde, Ouu); }
    );

    const auto Ouu_exact = Kokkos::View<double[6][6]>("Ouu");
    Kokkos::deep_copy(Ouu_exact, kExpectedOuu);
    const auto Ouu_mirror = Kokkos::create_mirror_view(Ouu);
    Kokkos::deep_copy(Ouu_mirror, Ouu);
    CompareWithExpected(Ouu_mirror, Ouu_exact, 1e-6);
}

TEST(CurvedBeamTests, CalculateOuuMatrixForCurvedBeam) {
    TestCalculateOuu();
}

void TestCalculatePuuForCurvedBeam() {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    const auto N_tilde = Kokkos::View<double[3][3]>("N_tilde");
    const auto Puu = Kokkos::View<double[6][6]>("Puu");

    // Set Cuu data
    const auto Cuu_mirror = Kokkos::create_mirror_view(Cuu);
    Kokkos::deep_copy(Cuu_mirror, kCurvedBeamCuu);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    // Set x0pupSS data
    const auto x0pupSS_mirror = Kokkos::create_mirror_view(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, kX0pupSS);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    // Set N_tilde data
    const auto N_tilde_mirror = Kokkos::create_mirror_view(N_tilde);
    Kokkos::deep_copy(N_tilde_mirror, kCurvedBeamN_tilde);
    Kokkos::deep_copy(N_tilde, N_tilde_mirror);

    Kokkos::parallel_for(
        "CalculatePuu", 1, KOKKOS_LAMBDA(size_t) { beams::CalculatePuu(Cuu, x0pupSS, N_tilde, Puu); }
    );

    const auto Puu_exact = Kokkos::View<double[6][6]>("Puu");
    Kokkos::deep_copy(Puu_exact, kExpectedPuu);
    const auto Puu_mirror = Kokkos::create_mirror_view(Puu);
    Kokkos::deep_copy(Puu_mirror, Puu);
    CompareWithExpected(Puu_mirror, Puu_exact, 1e-6);
}

TEST(CurvedBeamTests, CalculatePuuMatrixForCurvedBeam) {
    TestCalculatePuuForCurvedBeam();
}

void TestCalculateQuuForCurvedBeam() {
    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");
    const auto N_tilde = Kokkos::View<double[3][3]>("N_tilde");
    const auto Quu = Kokkos::View<double[6][6]>("Quu");

    // Set Cuu data
    const auto Cuu_mirror = Kokkos::create_mirror_view(Cuu);
    Kokkos::deep_copy(Cuu_mirror, kCurvedBeamCuu);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    // Set x0pupSS data
    const auto x0pupSS_mirror = Kokkos::create_mirror_view(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, kX0pupSS);
    Kokkos::deep_copy(x0pupSS, x0pupSS_mirror);

    // Set N_tilde data
    const auto N_tilde_mirror = Kokkos::create_mirror_view(N_tilde);
    Kokkos::deep_copy(N_tilde_mirror, kCurvedBeamN_tilde);
    Kokkos::deep_copy(N_tilde, N_tilde_mirror);

    Kokkos::parallel_for(
        "CalculateQuu", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateQuu(Cuu, x0pupSS, N_tilde, Quu); }
    );

    const auto Quu_exact = Kokkos::View<double[6][6]>("Quu");
    Kokkos::deep_copy(Quu_exact, kExpectedQuu);
    const auto Quu_mirror = Kokkos::create_mirror_view(Quu);
    Kokkos::deep_copy(Quu_mirror, Quu);
    CompareWithExpected(Quu_mirror, Quu_exact, 1e-6);
}

TEST(CurvedBeamTests, CalculateQuuMatrixForCurvedBeam) {
    TestCalculateQuuForCurvedBeam();
}

TEST(CurvedBeamTests, IntegrateStiffnessMatrixForCurvedBeam) {
    const auto qp_weights = get_qp_weights<kNumQPs>(kGaussQuadratureWeights);
    const auto qp_jacobian = get_qp_jacobian<kNumQPs>(kExpectedJacobians);
    const auto shape_interp = get_shape_interp<kNumNodes, kNumQPs>(kInterpWeightsFlat);
    const auto shape_deriv = get_shape_interp_deriv<kNumNodes, kNumQPs>(kDerivWeightsFlat);

    const auto qp_Kuu = get_qp_Kuu<kNumQPs>(kKuu);
    const auto qp_Puu = get_qp_Puu<kNumQPs>(kPuu);
    const auto qp_Cuu = get_qp_Cuu<kNumQPs>(kCuu);
    const auto qp_Ouu = get_qp_Ouu<kNumQPs>(kOuu);
    const auto qp_Quu = get_qp_Quu<kNumQPs>(kQuu);

    const auto stiffness_matrix_terms =
        Kokkos::View<double[kNumNodes][kNumNodes][6][6]>("stiffness_matrix_terms");

    constexpr auto simd_width = Kokkos::Experimental::native_simd<double>::size();
    constexpr auto extra_component = kNumNodes % simd_width == 0U ? 0U : 1U;
    constexpr auto simd_nodes = kNumNodes / simd_width + extra_component;
    const auto policy = Kokkos::RangePolicy(0, kNumNodes * simd_nodes);
    Kokkos::parallel_for(
        "IntegrateStiffnessMatrixElement", policy,
        beams::IntegrateStiffnessMatrixElement{
            0U, kNumNodes, kNumQPs, qp_weights, qp_jacobian, shape_interp, shape_deriv, qp_Kuu,
            qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, stiffness_matrix_terms
        }
    );

    const auto stiffness_matrix_exact =
        Kokkos::View<double[kNumNodes][kNumNodes][6][6], Kokkos::HostSpace>::const_type(
            kExpectedStiffnessMatrix.data()
        );

    const auto stiffness_matrix_terms_mirror = Kokkos::create_mirror_view(stiffness_matrix_terms);
    Kokkos::deep_copy(stiffness_matrix_terms_mirror, stiffness_matrix_terms);
    CompareWithExpected(stiffness_matrix_terms_mirror, stiffness_matrix_exact, 1e-7);
}

}  // namespace openturbine::tests::curved_beam
