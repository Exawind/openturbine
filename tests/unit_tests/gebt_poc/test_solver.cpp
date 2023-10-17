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
        Interpolate(generalized_coords, shape_function), {1.5, 2.5, 3.5, 0., 0., 1., 3.}
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
    auto rotation_0 = gen_alpha_solver::create_matrix({
        {1., 2., 3.},  // row 1
        {4., 5., 6.},  // row 2
        {7., 8., 9.}   // row 3
    });
    auto rotation = gen_alpha_solver::create_matrix({
        {1., 0., 0.},  // row 1
        {0., 1., 0.},  // row 2
        {0., 0., 1.}   // row 3
    });
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

TEST(SolverTest, CalculateElasticForces) {
    auto sectional_strain = Kokkos::View<double*>("sectional_strain", 6);
    auto populate_sectional_strain = KOKKOS_LAMBDA(size_t) {
        sectional_strain(0) = 1.1;
        sectional_strain(1) = 2.2;
        sectional_strain(2) = 3.3;
        sectional_strain(3) = 1.;
        sectional_strain(4) = 1.;
        sectional_strain(5) = 1.;
    };
    Kokkos::parallel_for(1, populate_sectional_strain);

    auto rotation = gen_alpha_solver::create_matrix({
        {1., 2., 3.},  // row 1
        {4., 5., 6.},  // row 2
        {7., 8., 9.}   // row 3
    });

    auto position_vector_derivatives = Kokkos::View<double*>("position_vector_derivatives", 7);
    auto populate_position_vector_derivatives = KOKKOS_LAMBDA(size_t) {
        position_vector_derivatives(0) = 1.;
        position_vector_derivatives(1) = 2.;
        position_vector_derivatives(2) = 3.;
        position_vector_derivatives(3) = 1.;
        position_vector_derivatives(4) = 0.;
        position_vector_derivatives(5) = 0.;
        position_vector_derivatives(6) = 0.;
    };
    Kokkos::parallel_for(1, populate_position_vector_derivatives);

    auto gen_coords_derivatives = Kokkos::View<double*>("gen_coords_derivatives", 7);
    auto populate_gen_coords_derivatives = KOKKOS_LAMBDA(size_t) {
        gen_coords_derivatives(0) = 0.1;
        gen_coords_derivatives(1) = 0.2;
        gen_coords_derivatives(2) = 0.3;
        gen_coords_derivatives(3) = 1.;
        gen_coords_derivatives(4) = 0.;
        gen_coords_derivatives(5) = 0.;
        gen_coords_derivatives(6) = 0.;
    };
    Kokkos::parallel_for(1, populate_gen_coords_derivatives);

    auto stiffness = gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    });
    auto elastic_forces = CalculateElasticForces(
        sectional_strain, rotation, position_vector_derivatives, gen_coords_derivatives, stiffness
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        elastic_forces, {-197.6, -395.2, -592.8, -790.4, -988., -1185.6, 0., 0., 0., 0., 0., 0.}
    );
}

TEST(SolverTest, CalculateStaticResidualZeroValues) {
    // 5 nodes on a line
    auto position_vectors = gen_alpha_solver::create_vector({
        0., 0., 0., 1., 0., 0., 0.,  // node 1
        1., 0., 0., 1., 0., 0., 0.,  // node 2
        2., 0., 0., 1., 0., 0., 0.,  // node 3
        3., 0., 0., 1., 0., 0., 0.,  // node 4
        4., 0., 0., 1., 0., 0., 0.   // node 5
    });

    // zero displacement and rotation
    auto generalized_coords = gen_alpha_solver::create_vector({
        0., 0., 0., 0., 0., 0., 0.,  // node 1
        0., 0., 0., 0., 0., 0., 0.,  // node 2
        0., 0., 0., 0., 0., 0., 0.,  // node 3
        0., 0., 0., 0., 0., 0., 0.,  // node 4
        0., 0., 0., 0., 0., 0., 0.   // node 5
    });

    // identity stiffness matrix
    auto stiffness = gen_alpha_solver::create_matrix({
        {1., 0., 0., 0., 0., 0.},  // row 1
        {0., 1., 0., 0., 0., 0.},  // row 2
        {0., 0., 1., 0., 0., 0.},  // row 3
        {0., 0., 0., 1., 0., 0.},  // row 4
        {0., 0., 0., 0., 1., 0.},  // row 5
        {0., 0., 0., 0., 0., 1.}   // row 6
    });

    // 7-point Gauss-Legendre quadrature
    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto residual =
        CalculateStaticResidual(position_vectors, generalized_coords, stiffness, quadrature);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        residual,
        // residuals expected to be all zeros
        {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}
    );
}

TEST(SolverTest, CalculateStaticResidualNonZeroValues) {
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
        generalized_coords(15) = -0.0125;
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

    auto residual =
        CalculateStaticResidual(position_vectors, generalized_coords, stiffness, quadrature);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        residual,
        {
            -0.11122699207269554,   // node 1, dof 1
            -0.16148820316250462,   // node 1, dof 2
            -0.304394950802795,     // node 1, dof 3
            -0.4038973586442003,    // node 1, dof 4
            -0.29270280250404546,   // node 1, dof 5
            -0.6838980603426406,    // node 1, dof 6
            -0.05267755488345887,   // node 2, dof 1
            -0.09882114478157883,   // node 2, dof 2
            -0.10319848820842535,   // node 2, dof 3
            -0.046921581717388396,  // node 2, dof 4
            0.20028020169482658,    // node 2, dof 5
            -0.49374296507736715,   // node 2, dof 6
            0.11946039620134324,    // node 3, dof 1
            0.1434326233631381,     // node 3, dof 2
            0.2789717998424498,     // node 3, dof 3
            0.28818850009642966,    // node 3, dof 4
            0.9034841540190914,     // node 3, dof 5
            0.21317918283700327,    // node 3, dof 6
            0.08547342278468156,    // node 4, dof 1
            0.2977996717461346,     // node 4, dof 2
            0.30720135556692413,    // node 4, dof 3
            0.34617026813588375,    // node 4, dof 4
            0.7528946278642024,     // node 4, dof 5
            0.5926242286597619,     // node 4, dof 6
            -0.04102927202987112,   // node 5, dof 1
            -0.18092294716519058,   // node 5, dof 2
            -0.17857971639815556,   // node 5, dof 3
            -0.06311517325798338,   // node 5, dof 4
            -0.5617067154657261,    // node 5, dof 5
            -0.259985875498865      // node 5, dof 6
        }
    );
}

TEST(SolverTest, CalculateIterationMatrixComponents) {
    auto elastic_force_fc = Kokkos::View<double*>("elastic_force_fc", 6);
    auto populate_elastic_force_fc = KOKKOS_LAMBDA(size_t) {
        elastic_force_fc(0) = 0.1023527958818833;
        elastic_force_fc(1) = 0.1512321779691288;
        elastic_force_fc(2) = 0.2788924951018168;
        elastic_force_fc(3) = 0.4003985306163255;
        elastic_force_fc(4) = 0.3249298550145402;
        elastic_force_fc(5) = 0.5876343707088096;
    };
    Kokkos::parallel_for(1, populate_elastic_force_fc);

    auto position_vector_derivatives = Kokkos::View<double*>("position_vector_derivatives", 7);
    auto populate_position_vector_derivatives = KOKKOS_LAMBDA(size_t) {
        position_vector_derivatives(0) = 0.924984344499876;
        position_vector_derivatives(1) = -0.3417491071948322;
        position_vector_derivatives(2) = 0.16616711516322974;
        position_vector_derivatives(3) = 0.023197240723436388;
        position_vector_derivatives(4) = 0.0199309451611758;
        position_vector_derivatives(5) = 0.0569650074322926;
    };
    Kokkos::parallel_for(1, populate_position_vector_derivatives);

    auto gen_coords_derivatives = Kokkos::View<double*>("gen_coords_derivatives", 7);
    auto populate_gen_coords_derivatives = KOKKOS_LAMBDA(size_t) {
        gen_coords_derivatives(0) = 0.0009414876868372689;
        gen_coords_derivatives(1) = -0.0009055519814222231;
        gen_coords_derivatives(2) = 0.000948674827920281;
        gen_coords_derivatives(3) = -0.000011768592509980857;
        gen_coords_derivatives(4) = 0.009249835939573452;
        gen_coords_derivatives(5) = 0.;
        gen_coords_derivatives(6) = 0.;
    };
    Kokkos::parallel_for(1, populate_gen_coords_derivatives);

    auto stiffness = gen_alpha_solver::create_matrix(
        {{1.3197900000789533, 1.9500660871457987, 3.596184383250417, 5.162946182374572,
          4.189813963364337, 7.577262149821259},
         {1.9500660871457987, 2.881335473074227, 5.313570498700886, 7.628551708533343,
          6.190692550267803, 11.19584324089132},
         {3.5961843832504163, 5.313570498700887, 9.798939314254936, 14.068076308736293,
          11.416470455810389, 20.64664212439219},
         {5.16294618237457, 7.628551708533343, 14.068076308736295, 20.197162639890838,
          16.390322707186375, 29.641834448609774},
         {4.189813963364336, 6.190692550267804, 11.416470455810385, 16.39032270718637,
          13.301010802137164, 24.054825962838},
         {7.577262149821259, 11.19584324089132, 20.64664212439219, 29.64183444860977,
          24.054825962838, 43.50305858028866}}
    );

    auto O_P_Q_matrices = CalculateIterationMatrixComponents(
        elastic_force_fc, position_vector_derivatives, gen_coords_derivatives, stiffness
    );
    auto o_matrix = Kokkos::View<double**>("o_matrix", 6, 6);
    auto populate_o_matrix = KOKKOS_LAMBDA(size_t i) {
        for (size_t j = 0; j < 6; ++j) {
            o_matrix(i, j) = O_P_Q_matrices(i, j);
        }
    };
    Kokkos::parallel_for(6, populate_o_matrix);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        o_matrix,
        {
            {0., 0., 0., 1.5581361688659614, 3.3881347643742066, -2.409080935189973},  // row 1
            {0., 0., 0., 2.023343846951859, 4.594085351204066, -3.233749380295583},    // row 2
            {0., 0., 0., 4.396865920548022, 8.369759049055988, -6.152221520070027},    // row 3
            {0., 0., 0., 6.095343338095462, 12.750819804192329, -9.15751050858455},    // row 4
            {0., 0., 0., 4.358834898453007, 9.870620837027674, -6.767382896430996},    // row 5
            {0., 0., 0., 9.270600163100875, 17.450580610135344, -12.962904649290419}   // row 6
        }
    );

    auto p_matrix = Kokkos::View<double**>("p_matrix", 6, 6);
    auto populate_p_matrix = KOKKOS_LAMBDA(size_t i) {
        for (size_t j = 0; j < 6; ++j) {
            p_matrix(i, j) = O_P_Q_matrices(i + 6, j);
        }
    };
    Kokkos::parallel_for(6, populate_p_matrix);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        p_matrix,
        {
            {0., 0., 0., 0., 0., 0.},  // row 1
            {0., 0., 0., 0., 0., 0.},  // row 2
            {0., 0., 0., 0., 0., 0.},  // row 3
            {1.558136168865961, 2.023343846951859, 4.396865920548022, 6.095343338095461,
             4.946469269161818, 8.945670308086335},  // row 4
            {3.388134764374206, 4.594085351204066, 8.369759049055988, 12.163185433483518,
             9.870620837027678, 17.85097914075167},  // row 5
            {-2.409080935189973, -3.233749380295583, -6.152221520070027, -8.83258065357001,
             -7.16778142704732, -12.962904649290419}  // row 6
        }
    );

    auto q_matrix = Kokkos::View<double**>("q_matrix", 6, 6);
    auto populate_q_matrix = KOKKOS_LAMBDA(size_t i) {
        for (size_t j = 0; j < 6; ++j) {
            q_matrix(i, j) = O_P_Q_matrices(i + 12, j);
        }
    };
    Kokkos::parallel_for(6, populate_q_matrix);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        q_matrix,
        {
            {0., 0., 0., 0., 0., 0.},                                                  // row 1
            {0., 0., 0., 0., 0., 0.},                                                  // row 2
            {0., 0., 0., 0., 0., 0.},                                                  // row 3
            {0., 0., 0., 1.844739298856163, 3.6356811370948883, -2.648497950457901},   // row 4
            {0., 0., 0., 3.8107825797230066, 7.1835652949545645, -5.293905367130955},  // row 5
            {0., 0., 0., -2.4073689531817264, -5.414742464880276, 3.8196948928089887}  // row 6

        }
    );
}

}  // namespace openturbine::gebt_poc::tests
