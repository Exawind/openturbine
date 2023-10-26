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

TEST(SolverTest, CalculateStaticIterationMatrixNonZeroValues) {
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

    auto iteration_matrix =
        CalculateStaticIterationMatrix(position_vectors, generalized_coords, stiffness, quadrature);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        iteration_matrix,
        {
            {1.7426373153996615,    2.59647015753046,      4.707627287468053,
             6.038991101730265,     4.008087516017232,     11.042058180068036,
             -1.9107725405221,      -2.891601845664942,    -5.172589353477725,
             -8.24040116031147,     -8.885129183012236,    -9.109936523704656,
             0.16839237294888507,   0.32034260596315434,   0.5083759354260828,
             0.7002188397894634,    1.7008676977600392,    0.40797378558670233,
             0.009820953060807937,  -0.007643644633245572, -0.036225766239858834,
             0.03849135600730806,   -0.3379478887198642,   0.1531503920189263,
             -0.010078100887250724, -0.017567273195419862, -0.007188103176541685,
             -0.027052952935513356, 0.03605103501770304,   -0.07055987920393175},  // row 1
            {2.5964701575304594,   3.8959604702875668,  7.0247233493506815,    9.14881391596337,
             6.2885447766529925,   16.276827006497694,  -2.891601845664942,    -4.456879334746913,
             -7.856342146183571,   -12.204797359480803, -13.271915902837236,   -14.134312887958405,
             0.32034260596315445,  0.7357931769365248,  1.0221865650289446,    1.1769936017329425,
             3.2531012676210684,   1.109377347151663,   -0.007643644633245614, -0.22840700790913063,
             -0.24271829195548203, 0.19221498430624556, -1.3197343077783255,   0.05722700038390871,
             -0.01756727319541985, 0.05353269543195904, 0.05215052375944118,   -0.1674893781988516,
             0.3780115113889654,   -0.08709007043932959},  // row 2
            {4.707627287468053,     7.024723349350682,    12.727552725906111,
             16.226821872479228,    11.27510066330126,    29.652376753430037,
             -5.172589353477725,    -7.856342146183572,   -14.028837027587453,
             -22.4178123138474,     -23.338237228631066,  -25.108384849603574,
             0.5083759354260828,    1.022186565028945,    1.56080180742091,
             2.1837033266476644,    4.618781757294458,    1.6809649335997077,
             -0.036225766239858834, -0.24271829195548225, -0.34274684943690953,
             -0.06279730043094356,  -1.5322054831433167,  -0.0018067731109545626,
             -0.007188103176541713, 0.052150523759441236, 0.0832293436973709,
             -0.07487070109280838,  0.42179048599121083,  -0.06140459364264439},  // row 3
            {6.038991101730264,    9.14881391596337,    16.226821872479228,   20.99517520456911,
             14.216244532372718,   38.06247387732293,   -6.532927723513471,   -10.10982743205247,
             -17.57205100464745,   -28.345333783935736, -30.61665881115424,   -30.964400512827375,
             0.5021184429087715,   1.1459429808029447,  1.5225923952079983,   2.1916842030628274,
             5.646331443511907,    1.094320092117063,   0.0210944308906814,   -0.2375225624356614,
             -0.22749454669643102, 0.14250588540100767, -1.8156157014655605,  0.43254071705774977,
             -0.02927625201622907, 0.0525930977218374,  0.050131283656690395, -0.16554150912102428,
             0.5346932885766651,   -0.20157769468028924},  // row 4
            {4.008087516017231,    6.288544776652993,    11.275100663301254,   14.866384196896023,
             10.746955705485718,   25.480462079720006,   -4.046386886695246,   -6.723299672388312,
             -11.68122723784334,   -18.127110109606015,  -22.20855089132184,   -19.590015245609955,
             -0.22253351105356894, 0.352504719044014,    0.07045204933299404,  -1.0608119568705072,
             3.9651386518362615,   -1.8416915851067772,  0.4558431616616395,   0.23211261719546483,
             0.6278969914570072,   2.2801136150462673,   -0.8295156118687896,  2.2769931129236247,
             -0.1950102799300468,  -0.14986244050414743, -0.29222246624789017, -0.9713122229120642,
             0.048812347049211446, -0.9215287733730968},  // row 5
            {11.042058180068036,  16.276827006497694,  29.652376753430037,   37.77988196207761,
             25.870374834536637,  69.33253483824618,   -12.42645780380891,   -18.543468767664347,
             -33.36447451685049,  -53.08502778582895,  -53.71119748392103,   -60.718619528253924,
             1.6897609575116572,  2.9805973200888567,  4.7796618790574525,   7.04181124025937,
             11.826078965983383,  6.494725294012317,   -0.39737955797870905, -1.0183437677379534,
             -1.5362567729681178, -1.3828110284290225, -4.657394193782189,   -1.5277508499744605,
             0.0920182242079475,  0.30438820881578144, 0.4686926573311839,   0.2446754270680026,
             1.463715897335318,   0.39328414879978785},  // row 6
            {-1.9107725405220999,    -2.8916018456649417,  -5.172589353477724,
             -6.532927723513473,     -4.046386886695245,   -12.42645780380891,
             3.145604574476188,      4.9276262363782815,   8.367066572977107,
             12.630567925449052,     11.214605581048524,   17.178200371539656,
             -1.3201660304884628,    -2.238026482398668,   -3.5214037638100923,
             -4.70019732422782,      -9.16893223365465,    -4.673539838833161,
             0.08713383599968194,    0.19276584528716256,  0.3745180960093597,
             0.13429252645069328,    1.5278834844942542,   0.036297855943415336,
             -0.0017998394653094482, 0.009236246398162729, -0.04759155169865367,
             0.021853714156374854,   -0.2312871939122656,  0.07434504782142828},  // row 7
            {-2.8916018456649417, -4.456879334746912,   -7.85634214618357,    -10.109827432052468,
             -6.72329967238831,   -18.543468767664347,  4.927626236378283,    7.95292503874701,
             13.187247027289265,  19.733717710956814,   18.197058260710673,   26.98988205646294,
             -2.2380264823986686, -4.205990417520162,   -6.194694664577004,   -7.55175802780435,
             -16.12591025931164,  -8.773413661563218,   0.19276584528716278,  0.9134505010685474,
             1.1022937318646937,  -0.22881059555374272, 4.700677859684006,    0.5092498133185237,
             0.0092362463981627,  -0.20350578754848359, -0.23850394839339117, 0.4179705125312473,
             -1.2600516608002916, 0.09012877594374347},  // row 8
            {-5.172589353477724,   -7.856342146183571,   -14.028837027587453, -17.572051004647445,
             -11.68122723784334,   -33.364474516850485,  8.367066572977107,   13.187247027289265,
             22.34996025998263,    33.63897669105289,    30.384413157639308,  45.62624166558231,
             -3.521403763810093,   -6.194694664577005,   -9.545140599873742,  -12.874504668600151,
             -23.54741514553661,   -13.622624542210259,  0.3745180960093599,  1.1022937318646937,
             1.607673134237903,    0.6812762704458505,   5.695762301403018,   1.1267602306614812,
             -0.04759155169865367, -0.23850394839339129, -0.3836557667593537, 0.11974283650902451,
             -1.4774593579043032,  -0.10796648339234682},  // row 9
            {-8.240401160311468,   -12.204797359480802,  -22.4178123138474,   -28.34533378393573,
             -18.193208918329454,  -53.06376917847401,   12.63056792544905,   19.733717710956814,
             33.638976691052896,   51.493967690344455,   46.90328587778654,   67.77231966855746,
             -4.852104314892383,   -8.75544738874555,    -12.81734792241342,  -17.750630242387018,
             -33.237140006029406,  -17.799536816843187,  0.5538549411814104,  1.629685486351828,
             2.161000244865881,    1.322524680958956,    8.144846650073685,   1.44051709142383,
             -0.09191739142662317, -0.40315844908230325, -0.5648166996579906, -0.012357897527162598,
             -2.284068525204252,   -0.20264378547056217},  // row 10
            {-8.885129183012234,  -13.271915902837236, -23.33823722863106,  -30.5505600024308,
             -22.20855089132184,  -53.732737778362015, 11.214605581048522,  18.197058260710673,
             30.384413157639305,  47.27614828931359,   51.161162886441495,  55.740391304004035,
             -1.8236377262525738, -5.081616715827775,  -6.780277779436084,  -5.650401084022256,
             -25.45234196330441,  -5.378858721931259,  -0.9745259271598582, -0.16543310737909156,
             -0.8583579620957495, -5.481301072971734,  4.426106460406455,   -5.149099669805337,
             0.468687255376128,   0.32190746533341086, 0.5924598125235465,  2.4765002794112108,
             -0.6008236539414016, 2.266273205516324},  // row 11
            {-9.109936523704654,   -14.134312887958403, -25.108384849603567, -30.98565912018229,
             -19.56847495116897,   -60.71861952825391,  17.178200371539656,  26.98988205646294,
             45.62624166558231,    68.00519794214654,   55.76003167634937,   97.24458430454344,
             -9.368980369500125,   -15.563005209572342, -24.51501109502956,  -34.05211930871497,
             -55.649759674223496,  -37.739070769888684, 1.6926643530786167,  3.7425550007605795,
             5.573308782993863,    4.626693108608469,   15.87405951058577,   6.583258031733774,
             -0.39194783141349726, -1.035118959692777,  -1.5761545039430511, -0.7324804095970743,
             -4.6436907490009585,  -1.6307025794178165},  // row 12
            {0.16839237294888512,  0.3203426059631544,  0.508375935426083,   0.502118442908772,
             -0.22253351105356842, 1.6897609575116577,  -1.3201660304884628, -2.238026482398668,
             -3.5214037638100923,  -4.852104314892383,  -1.8236377262525747, -9.368980369500123,
             1.8757951625167895,   3.6169492489069475,  5.125635286103051,   7.539445729706001,
             9.817887675122382,    9.618770963758001,   -0.8289134589918878, -1.9089866756189648,
             -2.441527844463806,   -1.7176320452568734, -7.398094836214979,  -2.9998624080627154,
             0.10489195401467293,  0.20972130314752496, 0.32892038674475554, 0.11630496644647989,
             1.0186918500857451,   0.31369871046809245},  // row 13
            {0.3203426059631545,  0.7357931769365247,  1.0221865650289446,  1.1459429808029449,
             0.3525047190440142,  2.980597320088857,   -2.2380264823986686, -4.205990417520162,
             -6.194694664577004,  -8.75544738874555,   -5.081616715827775,  -15.563005209572342,
             3.6169492489069475,  7.903456169115546,   10.374983176861708,  14.69948308655054,
             20.30611770802354,   20.067107679890636,  -1.9089866756189646, -5.3193561394168425,
             -6.2178985872497545, -3.0406200223194966, -20.156229949949,    -7.469561286563587,
             0.209721303147525,   0.8860972108849194,  1.0154235099360882,  -0.4090344241069154,
             3.9045254226141513,  0.7410158288528677},  // row 14
            {0.508375935426083,   1.022186565028945,  1.56080180742091,    1.5225923952079978,
             0.07045204933299526, 4.779661879057453,  -3.5214037638100923, -6.194694664577004,
             -9.54514059987374,   -12.81734792241342, -6.780277779436084,  -24.51501109502956,
             5.125635286103051,   10.374983176861708, 14.30397859261769,   20.752306421105928,
             27.37218375616062,   27.20204628610169,  -2.441527844463806,  -6.2178985872497545,
             -7.613134519927008,  -4.850316746107467, -23.29379295015963,  -9.580902730446805,
             0.32892038674475554, 1.0154235099360887, 1.293494719762121,   0.021217210337167836,
             4.372912382666064,   1.2007716975002034},  // row 15
            {0.7002188397894625,  1.176993601732943,   2.1837033266476658, 2.1916842030628274,
             -1.0196109354179494, 7.04421754758911,    -4.700197324227819, -7.5517580278043495,
             -12.87450466860015,  -17.750630242387018, -5.704267378622363, -34.09060113547666,
             7.539445729706001,   14.699483086550538,  20.752306421105924, 31.740352515872196,
             35.958823704581874,  41.417162631432625,  -4.294716676722476, -10.303098521021395,
             -12.563354129166566, -10.399410425616647, -36.39005335407397, -17.354623107007647,
             0.7552494314548184,  1.9783798605422414,  2.5018490500130897, 1.2432244777733206,
             7.563200016916799,   3.0709683741277534},  // row 16
            {1.7008676977600392,   3.2531012676210693, 4.618781757294456,  5.60513042205935,
             3.965138651836263,    11.83349008985475,  -9.16893223365465,  -16.12591025931164,
             -23.547415145536604,  -33.18327371142931, -25.45234196330441, -55.64587312770359,
             9.81788767512238,     20.306117708023535, 27.37218375616062,  35.78064597253502,
             65.7072369330199,     44.52884208924324,  -2.205941956979065, -8.1131446729609,
             -9.143235758148636,   0.3085290974287407, -36.16633242622093, -6.876053965369434,
             -0.14388118224872204, 0.6798359566278958, 0.6996853902301177, -2.9501707154220225,
             5.430516555908657,    -1.4109593025763258},  // row 17
            {0.4079737855867021,  1.1093773471516633,  1.6809649335997086,  1.091913784787325,
             -1.8491027089781429, 6.494725294012318,   -4.673539838833161,  -8.773413661563218,
             -13.622624542210257, -17.761054990081504, -5.38274526845116,   -37.739070769888684,
             9.618770963758003,   20.067107679890636,  27.20204628610169,   42.18616961591505,
             44.27674807544984,   56.81957249492407,   -6.442253638976067,  -15.426816949106797,
             -19.046903375969308, -14.759720690946796, -54.481216799677206, -26.47080633052632,
             1.089048728464506,   3.0237455836276768,  3.78651669847812,    1.3473798002953161,
             11.51010471114978,   4.555729191279031},  // row 18
            {0.009820953060807923, -0.007643644633245572, -0.03622576623985889, 0.021094430890681262,
             0.4558431616616394,   -0.3973795579787095,   0.08713383599968205,  0.1927658452871626,
             0.3745180960093599,   0.5538549411814108,    -0.9745259271598579,  1.6926643530786167,
             -0.8289134589918878,  -1.9089866756189648,   -2.441527844463806,   -4.294716676722476,
             -2.205941956979065,   -6.442253638976068,    1.1762832337427815,   3.452992268983921,
             3.9178883693849302,   4.1257709163218745,    10.042685580316537,   6.696317224794386,
             -0.4443245638113831,  -1.7291277940188745,   -1.8146528546906247,  -0.7546727439712445,
             -5.466019916720384,   -2.6896200840602633},  // row 19
            {-0.007643644633245614, -0.22840700790913068, -0.24271829195548203, -0.2375225624356613,
             0.2321126171954644,    -1.0183437677379534,  0.19276584528716278,  0.9134505010685476,
             1.1022937318646937,    1.6296854863518275,   -0.16543310737909067, 3.7425550007605803,
             -1.9089866756189648,   -5.319356139416843,   -6.2178985872497545,  -10.303098521021395,
             -8.1131446729609,      -15.426816949106797,  3.452992268983921,    12.528195400941211,
             13.272303649486778,    13.398386720099982,   34.30368747300814,    23.982324959045037,
             -1.7291277940188745,   -7.893882754683792,   -7.913980502146243,   -2.243226174038672,
             -25.44942414674964,    -10.919247449733696},  // row 20
            {-0.03622576623985889, -0.2427182919554823, -0.34274684943690964, -0.2274945466964313,
             0.6278969914570065,   -1.536256772968118,  0.3745180960093599,   1.1022937318646941,
             1.6076731342379034,   2.1610002448658814,  -0.8583579620957482,  5.573308782993863,
             -2.441527844463806,   -6.2178985872497545, -7.613134519927008,   -12.563354129166564,
             -9.143235758148638,   -19.046903375969308, 3.9178883693849302,   13.272303649486778,
             14.390423932921905,   15.026226488509955,  36.67468233268059,    25.786944672465623,
             -1.8146528546906247,  -7.913980502146242,  -8.042215697795896,   -2.41598873766201,
             -25.48302609697938,   -11.236222337336313},  // row 21
            {0.038491356007308325, 0.19221498430624512, -0.06279730043094373, 0.1425058854010074,
             2.272489281651,       -1.405760512602428,  0.13429252645069337,  -0.22881059555374228,
             0.6812762704458515,   1.3225246809589573,  -5.499498849716526,   4.707449602691521,
             -1.7176320452568743,  -3.040620022319497,  -4.8503167461074685,  -10.399410425616649,
             0.40399031030266297,  -14.905825136374935, 4.125770916321875,    13.398386720099982,
             15.026226488509955,   20.863259795431997,  31.550574217457935,   30.816754007996067,
             -2.580922753523007,   -10.321171086533015, -10.794388712417419,  -5.261344078815261,
             -31.9907653099729,    -16.361378078599465},  // row 22
            {-0.33794788871986414, -1.3197343077783255, -1.5322054831433163, -1.8079913680702928,
             -0.8295156118687905,  -4.653824222240083,  1.5278834844942546,  4.700677859684006,
             5.695762301403018,    8.163044426818482,   4.426106460406456,   15.850275561628035,
             -7.398094836214979,   -20.156229949949,    -23.293792950159627, -36.48551456694789,
             -36.16633242622093,   -54.4307400342937,   10.042685580316538,  34.30368747300814,
             36.67468233268059,    31.080825259772425,  102.97328307963957,  60.060974874864726,
             -3.8345263398759455,  -17.528401074964815, -17.544446200780662, -3.258596288189118,
             -58.25276852137649,   -22.914796369699204},  // row 23
            {0.15315039201892652, 0.05722700038390849, -0.0018067731109547847, 0.4554902012311539,
             2.273423141381519,   -1.5277508499744608, 0.03629785594341561,    0.5092498133185233,
             1.1267602306614826,  1.3597605973407805,  -5.125315720847599,     6.583258031733774,
             -2.9998624080627145, -7.469561286563588,  -9.580902730446805,     -17.208518661579504,
             -6.926530730752941,  -26.470806330526315, 6.696317224794386,      23.982324959045037,
             25.786944672465623,  31.464347931173748,  59.77425014554433,      50.65240181704923,
             -3.8859030646940167, -17.07924048618391,  -17.330995399569368,    -6.381511193480254,
             -53.758178953040655, -25.077911778430998},  // row 24
            {-0.010078100887250713,  -0.01756727319541985, -0.0071881031765416575,
             -0.02927625201622902,   -0.19501027993004685, 0.09201822420794765,
             -0.0017998394653094135, 0.0092362463981627,   -0.047591551698653645,
             -0.09191739142662325,   0.46868725537612826,  -0.3919478314134971,
             0.10489195401467293,    0.20972130314752496,  0.32892038674475554,
             0.7552494314548187,     -0.14388118224872215, 1.089048728464506,
             -0.4443245638113831,    -1.7291277940188745,  -1.8146528546906247,
             -2.5809227535230073,    -3.8345263398759455,  -3.8859030646940167,
             0.35131055014927126,    1.5277375176686108,   1.5405121228210685,
             0.6435670163039051,     4.642564225529215,    2.37213620497468},  // row 25
            {-0.01756727319541985, 0.0535326954319591,   0.05215052375944118, 0.052593097721837456,
             -0.1498624405041472,  0.30438820881578144,  0.0092362463981627,  -0.20350578754848359,
             -0.23850394839339117, -0.40315844908230314, 0.3219074653334104,  -1.0351189596927766,
             0.209721303147525,    0.8860972108849194,   1.0154235099360882,  1.9783798605422418,
             0.6798359566278966,   3.0237455836276754,   -1.7291277940188745, -7.893882754683792,
             -7.913980502146242,   -10.321171086533015,  -17.52840107496482,  -17.07924048618391,
             1.5277375176686108,   7.1577586359154175,   7.084910416844123,   2.4017794638132015,
             22.426938873546867,   10.175192915376444},  // row 26
            {-0.007188103176541685, 0.052150523759441236, 0.08322934369737092,   0.05013128365669045,
             -0.29222246624789006,  0.4686926573311839,   -0.047591551698653645, -0.2385039483933914,
             -0.3836557667593536,   -0.5648166996579904,  0.5924598125235465,    -1.5761545039430507,
             0.32892038674475554,   1.0154235099360887,   1.293494719762121,     2.5018490500130897,
             0.6996853902301186,    3.7865166984781196,   -1.8146528546906247,   -7.913980502146242,
             -8.042215697795896,    -10.794388712417419,  -17.54444620078066,    -17.330995399569368,
             1.5405121228210685,    7.084910416844123,    7.049147401095777,     2.3498993919086346,
             22.165782586226463,    10.204821716871127},  // row 27
            {-0.027052952935513495, -0.1674893781988515, -0.07487070109280822,
             -0.16554150912102422,  -0.9725484980652522, 0.2540708838154714,
             0.021853714156375006,  0.4179705125312474,  0.11974283650902501,
             -0.012357897527162237, 2.4937826059292365,  -0.7634156123791178,
             0.11630496644647983,   -0.4090344241069156, 0.021217210337168058,
             1.2432244777733212,    -2.99796520435825,   1.3950825956187416,
             -0.7546727439712443,   -2.2432261740386714, -2.4159887376620115,
             -5.261344078815261,    -3.2053601199487187, -6.398514462648277,
             0.643567016303905,     2.4017794638132015,  2.349899391908635,
             1.7771948702930391,    6.616540671163501,   4.244440713035745},  // row 28
            {0.036051035017703154, 0.37801151138896516, 0.4217904859912105,   0.5359295637298531,
             0.048812347049211446, 1.460290492535246,   -0.23128719391226576, -1.2600516608002912,
             -1.4774593579043036,  -2.3013508517222783, -0.6008236539413998,  -4.629534261494197,
             1.0186918500857454,   3.9045254226141504,  4.3729123826660645,   7.610994505853023,
             5.430516555908657,    11.484424761678028,  -5.466019916720384,   -25.449424146749635,
             -25.48302609697938,   -32.044001478213296, -58.25276852137648,   -53.72899620219303,
             4.642564225529215,    22.426938873546867,  22.165782586226467,   6.898014277331376,
             70.58304689618521,    31.542833040857882},  // row 29
            {-0.07055987920393181, -0.08709007043932926, -0.06140459364264428, -0.21097315142775763,
             -0.9181033685730251,  0.3932841487997876,   0.07434504782142862,  0.09012877594374324,
             -0.10796648339234682, -0.1717085826885174,  2.252116718009564,    -1.6307025794178165,
             0.31369871046809256,  0.7410158288528685,   1.2007716975002038,   3.023265578804328,
             -1.3852793531045773,  4.555729191279031,    -2.6896200840602633,  -10.919247449733694,
             -11.236222337336315,  -16.344374809431443,  -22.943979120546818,  -25.077911778430998,
             2.37213620497468,     10.175192915376442,   10.20482171687113,    3.6918937776908463,
             31.62018209819842,    15.063161009023737}  // row 30
        }
    );
}

TEST(SolverTest, ConstraintsResidualVector) {
    auto position_vectors = Kokkos::View<double*>("position_vectors", 7);
    auto populate_position_vector = KOKKOS_LAMBDA(size_t) {
        position_vectors(0) = 0.;
        position_vectors(1) = 0.;
        position_vectors(2) = 0.;
        position_vectors(3) = 0.9778215200524469;
        position_vectors(4) = -0.01733607539094763;
        position_vectors(5) = -0.09001900002195001;
        position_vectors(6) = -0.18831121859148398;
    };
    Kokkos::parallel_for(1, populate_position_vector);

    auto generalized_coords = Kokkos::View<double*>("generalized_coords", 7);
    auto populate_generalized_coords = KOKKOS_LAMBDA(size_t) {
        generalized_coords(0) = 0.1;
        generalized_coords(1) = 0.;
        generalized_coords(2) = 0.12;
        generalized_coords(3) = 0.9987502603949662;
        generalized_coords(4) = 0.049979169270678324;
        generalized_coords(5) = 0.;
        generalized_coords(6) = 0.;
    };
    Kokkos::parallel_for(1, populate_generalized_coords);

    auto constraints_residual = ConstraintsResidualVector(generalized_coords, position_vectors);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        constraints_residual, {0.1, 0., 0.12, 0.1, -0.01198, 0.1194}
    );
}

}  // namespace openturbine::gebt_poc::tests
