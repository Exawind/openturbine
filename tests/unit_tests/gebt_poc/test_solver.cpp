#include <gtest/gtest.h>

#include "src/gebt_poc/solver.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(SolverTest, UserDefinedQuadrature) {
    auto quadrature_points = std::vector<double>{
        -0.9491079123427585,  // point 1
        -0.7415311855993945,  // point 2
        -0.4058451513773972,  // point 3
        0.,                   // point 4
        0.4058451513773972,   // point 5
        0.7415311855993945,   // point 6
        0.9491079123427585    // point 7
    };
    auto quadrature_weights = std::vector<double>{
        0.1294849661688697,  // weight 1
        0.2797053914892766,  // weight 2
        0.3818300505051189,  // weight 3
        0.4179591836734694,  // weight 4
        0.3818300505051189,  // weight 5
        0.2797053914892766,  // weight 6
        0.1294849661688697   // weight 7
    };
    auto quadrature = UserDefinedQuadrature(quadrature_points, quadrature_weights);

    EXPECT_EQ(quadrature.GetNumberOfQuadraturePoints(), 7);
    EXPECT_EQ(quadrature.GetQuadraturePoints(), quadrature_points);
    EXPECT_EQ(quadrature.GetQuadratureWeights(), quadrature_weights);
}

TEST(SolverTest, CalculateInterpolatedValues) {
    auto generalized_coords = Kokkos::View<double*>("generalized_coords", 14);
    auto populate_generalized_coords = KOKKOS_LAMBDA(size_t) {
        // node 1
        generalized_coords(0) = 1.;
        generalized_coords(1) = 2.;
        generalized_coords(2) = 3.;
        generalized_coords(3) = 0.;
        generalized_coords(4) = -1.;
        generalized_coords(5) = -2.;
        generalized_coords(6) = -3.;
        // node 2
        generalized_coords(7) = 2.;
        generalized_coords(8) = 3.;
        generalized_coords(9) = 4.;
        generalized_coords(10) = 0.;
        generalized_coords(11) = 1.;
        generalized_coords(12) = 4.;
        generalized_coords(13) = 9.;
    };
    Kokkos::parallel_for(1, populate_generalized_coords);
    auto quadrature_pt = 0.;
    auto shape_function = gen_alpha_solver::create_vector(LagrangePolynomial(1, quadrature_pt));

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        Interpolate(generalized_coords, shape_function, quadrature_pt),
        {1.5, 2.5, 3.5, 0., 0., 1., 3.}
    );
}

TEST(SolverTest, CalculateCurvature) {
    auto rotation_matrix = gen_alpha_solver::RotationMatrix(
        0.8146397707387071, -0.4884001129794905, 0.31277367787652416, 0.45607520213614394,
        0.8726197541000288, 0.17472886066955512, -0.3582700851693625, 0.00030723936311904954,
        0.933617936672551
    );
    auto q = gen_alpha_solver::rotation_matrix_to_quaternion(rotation_matrix);

    auto gen_coords = Kokkos::View<double*>("gen_coords", 7);
    auto populate_gen_coords = KOKKOS_LAMBDA(size_t) {
        gen_coords(0) = 0.;
        gen_coords(1) = 0.;
        gen_coords(2) = 0.;
        gen_coords(3) = q.GetScalarComponent();
        gen_coords(4) = q.GetXComponent();
        gen_coords(5) = q.GetYComponent();
        gen_coords(6) = q.GetZComponent();
    };
    Kokkos::parallel_for(1, populate_gen_coords);

    auto gen_coords_derivative = Kokkos::View<double*>("gen_coords_derivative", 7);
    auto populate_gen_coords_derivative = KOKKOS_LAMBDA(size_t) {
        gen_coords_derivative(0) = 0.;
        gen_coords_derivative(1) = 0.;
        gen_coords_derivative(2) = 0.;
        gen_coords_derivative(3) = -0.0257045;
        gen_coords_derivative(4) = -0.0230032;
        gen_coords_derivative(5) = 0.030486;
        gen_coords_derivative(6) = 0.0694527;
    };
    Kokkos::parallel_for(1, populate_gen_coords_derivative);

    auto curvature = CalculateCurvature(gen_coords, gen_coords_derivative);

    EXPECT_NEAR(curvature(0), -0.03676700256944363, 1e-6);
    EXPECT_NEAR(curvature(1), 0.062023963818612256, 1e-6);
    EXPECT_NEAR(curvature(2), 0.15023478838786522, 1e-6);
}

TEST(SolverTest, CalculateSectionalStiffness) {
    auto rotation_0 = gen_alpha_solver::RotationMatrix(1., 2., 3., 4., 5., 6., 7., 8., 9.);
    auto rotation = gen_alpha_solver::RotationMatrix(1., 0., 0., 0., 1., 0., 0., 0., 1.);
    auto stiffness = StiffnessMatrix(gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    }));

    auto sectional_stiffness = CalculateSectionalStiffness(stiffness, rotation_0, rotation);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        sectional_stiffness,
        {
            {196., 448., 700., 448., 1078., 1708.},      // row 1
            {448., 1024., 1600., 1024., 2464., 3904.},   // row 2
            {700., 1600., 2500., 1600., 3850., 6100.},   // row 3
            {448., 1024., 1600., 1024., 2464., 3904.},   // row 4
            {1078., 2464., 3850., 2464., 5929., 9394.},  // row 5
            {1708., 3904., 6100., 3904., 9394., 14884.}  // row 6
        }
    );
}

TEST(SolverTest, CalculateStaticResidual) {
    auto position_vectors = Kokkos::View<double*>("position_vectors", 35);
    auto populate_position_vector = KOKKOS_LAMBDA(size_t) {
        // node 1
        position_vectors(0) = 0.;
        position_vectors(1) = 0.;
        position_vectors(2) = 0.;
        position_vectors(3) = 0.9778215200524469;
        position_vectors(4) = -0.01733607539094763;
        position_vectors(5) = -0.09001900002195001;
        position_vectors(6) = -0.18831121859148398;
        // node 2
        position_vectors(7) = 0.8633658232300573;
        position_vectors(8) = -0.25589826392541715;
        position_vectors(9) = 0.1130411210682743;
        position_vectors(10) = 0.9950113028068008;
        position_vectors(11) = -0.002883848832932071;
        position_vectors(12) = -0.030192109815745303;
        position_vectors(13) = -0.09504013471947484;
        // node 3
        position_vectors(14) = 2.5;
        position_vectors(15) = -0.25;
        position_vectors(16) = 0.;
        position_vectors(17) = 0.9904718430204884;
        position_vectors(18) = -0.009526411091536478;
        position_vectors(19) = 0.09620741150793366;
        position_vectors(20) = 0.09807604012323785;
        // node 4
        position_vectors(21) = 4.136634176769943;
        position_vectors(22) = 0.39875540678255983;
        position_vectors(23) = -0.5416125496397027;
        position_vectors(24) = 0.9472312341234699;
        position_vectors(25) = -0.049692141629315074;
        position_vectors(26) = 0.18127630174800594;
        position_vectors(27) = 0.25965858850765167;
        // node 5
        position_vectors(28) = 5.;
        position_vectors(29) = 1.;
        position_vectors(30) = -1.;
        position_vectors(31) = 0.9210746582719719;
        position_vectors(32) = -0.07193653093139739;
        position_vectors(33) = 0.20507529985516368;
        position_vectors(34) = 0.32309554437664584;
    };
    Kokkos::parallel_for(1, populate_position_vector);

    auto generalized_coords = Kokkos::View<double*>("generalized_coords", 35);
    auto populate_generalized_coords = KOKKOS_LAMBDA(size_t) {
        // node 1
        generalized_coords(0) = 0.;
        generalized_coords(1) = 0.;
        generalized_coords(2) = 0.;
        generalized_coords(3) = 1.;
        generalized_coords(4) = 0.;
        generalized_coords(5) = 0.;
        generalized_coords(6) = 0.;
        // node 2
        generalized_coords(7) = 0.0029816021788868583;
        generalized_coords(8) = -0.0024667594949430213;
        generalized_coords(9) = 0.0030845707156756256;
        generalized_coords(10) = 0.9999627302042724;
        generalized_coords(11) = 0.008633550973807838;
        generalized_coords(12) = 0.;
        generalized_coords(13) = 0.;
        // node 3
        generalized_coords(14) = 0.025;
        generalized_coords(15) = 0.0125;
        generalized_coords(16) = 0.027500000000000004;
        generalized_coords(17) = 0.9996875162757026;
        generalized_coords(18) = 0.024997395914712332;
        generalized_coords(19) = 0.;
        generalized_coords(20) = 0.;
        // node 4
        generalized_coords(21) = 0.06844696924968456;
        generalized_coords(22) = -0.011818954790771264;
        generalized_coords(23) = 0.07977257214146723;
        generalized_coords(24) = 0.9991445348823056;
        generalized_coords(25) = 0.04135454527402519;
        generalized_coords(26) = 0.;
        generalized_coords(27) = 0.;
        // node 5
        generalized_coords(28) = 0.1;
        generalized_coords(29) = 0.;
        generalized_coords(30) = 0.12;
        generalized_coords(31) = 0.9987502603949662;
        generalized_coords(32) = 0.049979169270678324;
        generalized_coords(33) = 0.;
        generalized_coords(34) = 0.;
    };
    Kokkos::parallel_for(1, populate_generalized_coords);

    auto stiffness = StiffnessMatrix(gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    }));

    auto quadrature_points =
        std::vector<double>{-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.,
                            0.4058451513773972,  0.7415311855993945,  0.9491079123427585};
    auto quadrature_weights = std::vector<double>{
        0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
        0.3818300505051189, 0.2797053914892766, 0.1294849661688697};
    auto quadrature = UserDefinedQuadrature(quadrature_points, quadrature_weights);

    // auto residual =
    //     CalculateStaticResidual(position_vectors, generalized_coords, stiffness, quadrature);

    // EXPECT_NEAR(residual(0), -0.111227, 1e-6);
    // EXPECT_NEAR(residual(1), -0.161488, 1e-6);
    // EXPECT_NEAR(residual(2), -0.304395, 1e-6);
    // EXPECT_NEAR(residual(3), -0.403897, 1e-6);
    // EXPECT_NEAR(residual(4), -0.292703, 1e-6);
    // EXPECT_NEAR(residual(5), -0.683898, 1e-6);
    // EXPECT_NEAR(residual(6), -0.0526776, 1e-6);
    // EXPECT_NEAR(residual(7), -0.0988211, 1e-6);
    // EXPECT_NEAR(residual(8), -0.103198, 1e-6);
    // EXPECT_NEAR(residual(9), -0.0469216, 1e-6);
    // EXPECT_NEAR(residual(10), 0.20028, 1e-6);
    // EXPECT_NEAR(residual(11), -0.493743, 1e-6);
    // EXPECT_NEAR(residual(12), 0.11946, 1e-6);
    // EXPECT_NEAR(residual(13), 0.143433, 1e-6);
    // EXPECT_NEAR(residual(14), 0.278972, 1e-6);
    // EXPECT_NEAR(residual(15), 0.288189, 1e-6);
    // EXPECT_NEAR(residual(16), 0.903484, 1e-6);
    // EXPECT_NEAR(residual(17), 0.213179, 1e-6);
    // EXPECT_NEAR(residual(18), 0.0854734, 1e-6);
    // EXPECT_NEAR(residual(19), 0.2978, 1e-6);
    // EXPECT_NEAR(residual(20), 0.307201, 1e-6);
    // EXPECT_NEAR(residual(21), 0.34617, 1e-6);
    // EXPECT_NEAR(residual(22), 0.752895, 1e-6);
    // EXPECT_NEAR(residual(23), 0.592624, 1e-6);
    // EXPECT_NEAR(residual(24), -0.0410293, 1e-6);
    // EXPECT_NEAR(residual(25), -0.180923, 1e-6);
    // EXPECT_NEAR(residual(26), -0.17858, 1e-6);
    // EXPECT_NEAR(residual(27), -0.0631152, 1e-6);
    // EXPECT_NEAR(residual(28), -0.561707, 1e-6);
    // EXPECT_NEAR(residual(29), -0.259986, 1e-6);
}

}  // namespace openturbine::gebt_poc::tests
