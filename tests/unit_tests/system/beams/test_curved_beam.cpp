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
    const auto node_position_rotation =
        CreateView<double[kNumElems][kNumNodes][7]>("node_position_rotation", kCurvedBeamNodes_data);
    const auto shape_derivative =
        CreateView<double[kNumElems][kNumNodes][kNumQPs]>("shape_derivative", kDerivWeightsFlat);
    const auto kNumNodes_per_elem =
        CreateView<size_t[kNumElems]>("number_of_nodes_per_element", std::array{kNumNodes});
    const auto kNumQPs_per_elem = CreateView<size_t[kNumElems]>(
        "number_of_quadrature_points_per_element", std::array{kNumQPs}
    );

    const auto qp_position_derivative =
        Kokkos::View<double[kNumElems][kNumQPs][3]>("position_derivative");
    const auto qp_jacobian = Kokkos::View<double[kNumElems][kNumQPs]>("jacobian");
    const auto calculate_jacobian =
        CalculateJacobian<Kokkos::DefaultExecutionSpace>{kNumNodes_per_elem,     kNumQPs_per_elem,       shape_derivative,
                          node_position_rotation, qp_position_derivative, qp_jacobian};
    Kokkos::parallel_for("calculate_jacobian", 1, calculate_jacobian);
    const auto qp_jacobian_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qp_jacobian);

    // Validate the calculated jacobians against expected values from Mathematica script
    for (size_t qp = 0; qp < kNumQPs; ++qp) {
        EXPECT_NEAR(qp_jacobian_mirror(0, qp), kExpectedJacobians[qp], kDefaultTolerance)
            << "Jacobian mismatch at quadrature point " << qp;
    }

    // Verify that position derivatives are unit vectors
    const auto qp_position_derivative_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qp_position_derivative);
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
    const auto Cuu = CreateView<double[6][6]>("Cuu", kCurvedBeamCuu);
    const auto strain = CreateView<double[6]>("strain", kCurvedBeamStrain);

    const auto Fc = Kokkos::View<double[6]>("Fc");
    Kokkos::parallel_for(
        "CalculateForceFc", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateForceFC<Kokkos::DefaultExecutionSpace>(Cuu, strain, Fc); }
    );

    const auto Fc_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Fc);
    CompareWithExpected(Fc_mirror, kExpectedFc, 1e-6);
}

TEST(CurvedBeamTests, CalculateForceFcForCurvedBeam) {
    TestCalculateForceFc();
}

void TestCalculateForceFd() {
    const auto x0pupSS = CreateView<double[3][3]>("x0pupSS", kX0pupSS);
    const auto Fc = CreateView<double[6]>("Fc", kExpectedFc);

    const auto Fd = Kokkos::View<double[6]>("Fd");
    Kokkos::parallel_for(
        "CalculateForceFd", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateForceFD<Kokkos::DefaultExecutionSpace>(x0pupSS, Fc, Fd); }
    );

    const auto Fd_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Fd);
    CompareWithExpected(Fd_mirror, kExpectedFd, 1e-6);
}

TEST(CurvedBeamTests, CalculateForceFdForCurvedBeam) {
    TestCalculateForceFd();
}

TEST(CurvedBeamTests, IntegrateResidualVectorForCurvedBeam) {
    const auto qp_weights = CreateView<double[kNumQPs]>("qp_weights", kGaussQuadratureWeights);
    const auto qp_jacobian = CreateView<double[kNumQPs]>("qp_jacobian", kExpectedJacobians);
    const auto shape_interp =
        CreateLeftView<double[kNumNodes][kNumQPs]>("shape_interp", kInterpWeightsFlat);
    const auto shape_deriv =
        CreateLeftView<double[kNumNodes][kNumQPs]>("shape_deriv", kDerivWeightsFlat);

    const auto node_FX = Kokkos::View<double[kNumNodes][6]>("node_FX");
    const auto qp_Fc = CreateView<double[kNumQPs][6]>("qp_Fc", kFc);
    const auto qp_Fd = CreateView<double[kNumQPs][6]>("qp_Fd", kFd);
    const auto qp_Fi = CreateView<double[kNumQPs][6]>("qp_Fi", kFi);
    const auto qp_Fe = Kokkos::View<double[kNumQPs][6]>("qp_Fe");
    const auto qp_Fg = Kokkos::View<double[kNumQPs][6]>("qp_Fg");

    const auto residual_vector_terms =
        Kokkos::View<double[kNumElems][kNumNodes][6]>("residual_vector_terms");

    Kokkos::parallel_for(
        "IntegrateResidualVectorElement", kNumNodes,
        beams::IntegrateResidualVectorElement<Kokkos::DefaultExecutionSpace>{
            0U, kNumQPs, qp_weights, qp_jacobian, shape_interp, shape_deriv, node_FX, qp_Fc, qp_Fd,
            qp_Fi, qp_Fe, qp_Fg, residual_vector_terms
        }
    );

    const auto residual_vector_exact =
        Kokkos::View<double[kNumElems][kNumNodes][6], Kokkos::HostSpace>::const_type(
            kExpectedResidualVector.data()
        );

    const auto residual_vector_terms_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), residual_vector_terms);
    CompareWithExpected(residual_vector_terms_mirror, residual_vector_exact, 1e-8);
}

void TestRotateSectionMatrixForCurvedBeam() {
    const auto xr = CreateView<double[4]>("xr", kCurvedBeamXr);
    const auto Cstar = CreateView<double[6][6]>("Cstar", kCurvedBeamCstar);

    const auto Cuu = Kokkos::View<double[6][6]>("Cuu");
    Kokkos::parallel_for(
        "RotateSectionMatrix", 1,
        KOKKOS_LAMBDA(size_t) { beams::RotateSectionMatrix(xr, Cstar, Cuu); }
    );

    const auto Cuu_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Cuu);
    CompareWithExpected(Cuu_mirror, kExpectedCuu, 1e-6);
}

TEST(CurvedBeamTests, CalculateRotatedStiffnessMatrixForCurvedBeam) {
    TestRotateSectionMatrixForCurvedBeam();
}

void TestCalculateOuu() {
    const auto Cuu = CreateView<double[6][6]>("Cuu", kCurvedBeamCuu);
    const auto x0pupSS = CreateView<double[3][3]>("x0pupSS", kX0pupSS);
    const auto M_tilde = CreateView<double[3][3]>("M_tilde", kCurvedBeamM_tilde);
    const auto N_tilde = CreateView<double[3][3]>("N_tilde", kCurvedBeamN_tilde);

    const auto Ouu = Kokkos::View<double[6][6]>("Ouu");
    Kokkos::parallel_for(
        "CalculateOuu", 1,
        KOKKOS_LAMBDA(size_t) { beams::CalculateOuu<Kokkos::DefaultExecutionSpace>(Cuu, x0pupSS, M_tilde, N_tilde, Ouu); }
    );

    const auto Ouu_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ouu);
    CompareWithExpected(Ouu_mirror, kExpectedOuu, 1e-6);
}

TEST(CurvedBeamTests, CalculateOuuMatrixForCurvedBeam) {
    TestCalculateOuu();
}

void TestCalculatePuuForCurvedBeam() {
    const auto Cuu = CreateView<double[6][6]>("Cuu", kCurvedBeamCuu);
    const auto x0pupSS = CreateView<double[3][3]>("x0pupSS", kX0pupSS);
    const auto N_tilde = CreateView<double[3][3]>("N_tilde", kCurvedBeamN_tilde);

    const auto Puu = Kokkos::View<double[6][6]>("Puu");
    Kokkos::parallel_for(
        "CalculatePuu", 1, KOKKOS_LAMBDA(size_t) { beams::CalculatePuu<Kokkos::DefaultExecutionSpace>(Cuu, x0pupSS, N_tilde, Puu); }
    );

    const auto Puu_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Puu);
    CompareWithExpected(Puu_mirror, kExpectedPuu, 1e-6);
}

TEST(CurvedBeamTests, CalculatePuuMatrixForCurvedBeam) {
    TestCalculatePuuForCurvedBeam();
}

void TestCalculateQuuForCurvedBeam() {
    const auto Cuu = CreateView<double[6][6]>("Cuu", kCurvedBeamCuu);
    const auto x0pupSS = CreateView<double[3][3]>("x0pupSS", kX0pupSS);
    const auto N_tilde = CreateView<double[3][3]>("N_tilde", kCurvedBeamN_tilde);

    const auto Quu = Kokkos::View<double[6][6]>("Quu");
    Kokkos::parallel_for(
        "CalculateQuu", 1, KOKKOS_LAMBDA(size_t) { beams::CalculateQuu<Kokkos::DefaultExecutionSpace>(Cuu, x0pupSS, N_tilde, Quu); }
    );

    const auto Quu_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Quu);
    CompareWithExpected(Quu_mirror, kExpectedQuu, 1e-6);
}

TEST(CurvedBeamTests, CalculateQuuMatrixForCurvedBeam) {
    TestCalculateQuuForCurvedBeam();
}

TEST(CurvedBeamTests, IntegrateStiffnessMatrixForCurvedBeam) {
    const auto qp_weights = CreateView<double[kNumQPs]>("qp_weights", kGaussQuadratureWeights);
    const auto qp_jacobian = CreateView<double[kNumQPs]>("qp_jacobian", kExpectedJacobians);
    const auto shape_interp =
        CreateLeftView<double[kNumNodes][kNumQPs]>("shape_interp", kInterpWeightsFlat);
    const auto shape_deriv =
        CreateLeftView<double[kNumNodes][kNumQPs]>("shape_deriv", kDerivWeightsFlat);

    const auto qp_Kuu = CreateView<double[kNumQPs][6][6]>("qp_Kuu", kKuu);
    const auto qp_Puu = CreateView<double[kNumQPs][6][6]>("qp_Puu", kPuu);
    const auto qp_Cuu = CreateView<double[kNumQPs][6][6]>("qp_Cuu", kCuu);
    const auto qp_Ouu = CreateView<double[kNumQPs][6][6]>("qp_Ouu", kOuu);
    const auto qp_Quu = CreateView<double[kNumQPs][6][6]>("qp_Quu", kQuu);

    const auto stiffness_matrix_terms =
        Kokkos::View<double[kNumNodes][kNumNodes][6][6]>("stiffness_matrix_terms");

    constexpr auto simd_width = Kokkos::Experimental::simd<double>::size();
    constexpr auto extra_component = kNumNodes % simd_width == 0U ? 0U : 1U;
    constexpr auto simd_nodes = kNumNodes / simd_width + extra_component;
    const auto policy = Kokkos::RangePolicy(0, kNumNodes * simd_nodes);
    Kokkos::parallel_for(
        "IntegrateStiffnessMatrixElement", policy,
        beams::IntegrateStiffnessMatrixElement<Kokkos::DefaultExecutionSpace>{
            0U, kNumNodes, kNumQPs, qp_weights, qp_jacobian, shape_interp, shape_deriv, qp_Kuu,
            qp_Puu, qp_Cuu, qp_Ouu, qp_Quu, stiffness_matrix_terms
        }
    );

    const auto stiffness_matrix_terms_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), stiffness_matrix_terms);
    CompareWithExpected(stiffness_matrix_terms_mirror, kExpectedStiffnessMatrix, 1e-7);
}

}  // namespace openturbine::tests::curved_beam
