#include "tests/unit_tests/gebt_poc/test_johnson_static_beam.h"

#include <gtest/gtest.h>

#include "src/gebt_poc/gen_alpha_2D.h"
#include "src/gebt_poc/solver.h"
#include "src/gebt_poc/static_beam_element.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

struct DefinePositionVector_5NodeBeamElement {
    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        position_vectors_(0, 0) = 0.;
        position_vectors_(0, 1) = 0.;
        position_vectors_(0, 2) = 0.;
        position_vectors_(0, 3) = 0.9778215200524469;
        position_vectors_(0, 4) = -0.01733607539094763;
        position_vectors_(0, 5) = -0.09001900002195001;
        position_vectors_(0, 6) = -0.18831121859148398;
        // node 2
        position_vectors_(1, 0) = 0.8633658232300573;
        position_vectors_(1, 1) = -0.25589826392541715;
        position_vectors_(1, 2) = 0.1130411210682743;
        position_vectors_(1, 3) = 0.9950113028068008;
        position_vectors_(1, 4) = -0.002883848832932071;
        position_vectors_(1, 5) = -0.030192109815745303;
        position_vectors_(1, 6) = -0.09504013471947484;
        // node 3
        position_vectors_(2, 0) = 2.5;
        position_vectors_(2, 1) = -0.25;
        position_vectors_(2, 2) = 0.;
        position_vectors_(2, 3) = 0.9904718430204884;
        position_vectors_(2, 4) = -0.009526411091536478;
        position_vectors_(2, 5) = 0.09620741150793366;
        position_vectors_(2, 6) = 0.09807604012323785;
        // node 4
        position_vectors_(3, 0) = 4.136634176769943;
        position_vectors_(3, 1) = 0.39875540678255983;
        position_vectors_(3, 2) = -0.5416125496397027;
        position_vectors_(3, 3) = 0.9472312341234699;
        position_vectors_(3, 4) = -0.049692141629315074;
        position_vectors_(3, 5) = 0.18127630174800594;
        position_vectors_(3, 6) = 0.25965858850765167;
        // node 5
        position_vectors_(4, 0) = 5.;
        position_vectors_(4, 1) = 1.;
        position_vectors_(4, 2) = -1.;
        position_vectors_(4, 3) = 0.9210746582719719;
        position_vectors_(4, 4) = -0.07193653093139739;
        position_vectors_(4, 5) = 0.20507529985516368;
        position_vectors_(4, 6) = 0.32309554437664584;
    }
    Kokkos::View<double[5][7]> position_vectors_;
};

StaticBeamLinearizationParameters create_test_static_beam_parameters() {
    auto position_vectors = Kokkos::View<double[5][7]>("position_vectors");
    Kokkos::parallel_for(1, DefinePositionVector_5NodeBeamElement{position_vectors});

    auto stiffness_matrix = StiffnessMatrix(gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    }));

    auto quadrature = UserDefinedQuadrature(
        std::vector<double>{
            -0.9491079123427585,  // point 1
            -0.7415311855993945,  // point 2
            -0.4058451513773972,  // point 3
            0.,                   // point 4
            0.4058451513773972,   // point 5
            0.7415311855993945,   // point 6
            0.9491079123427585    // point 7
        },
        std::vector<double>{
            0.1294849661688697,  // weight 1
            0.2797053914892766,  // weight 2
            0.3818300505051189,  // weight 3
            0.4179591836734694,  // weight 4
            0.3818300505051189,  // weight 5
            0.2797053914892766,  // weight 6
            0.1294849661688697   // weight 7
        }
    );
    return StaticBeamLinearizationParameters{position_vectors, stiffness_matrix, quadrature};
}

TEST(StaticBeamTest, CalculateTangentOperatorWithPhiAsZero) {
    auto psi = gen_alpha_solver::create_matrix({{0., 0., 0., 0., 0., 0.}});
    auto static_beam = create_test_static_beam_parameters();

    auto tangent_operator = Kokkos::View<double[6][6]>("tangent_operator");
    static_beam.TangentOperator(psi, 1., tangent_operator);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        tangent_operator,
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0.},  // row 3
            {0., 0., 0., 1., 0., 0.},  // row 4
            {0., 0., 0., 0., 1., 0.},  // row 5
            {0., 0., 0., 0., 0., 1.}   // row 6
        }
    );
}

TEST(StaticBeamTest, CalculateTangentOperatorWithPhiNotZero) {
    auto psi = gen_alpha_solver::create_matrix({{0., 0., 0., 1., 2., 3.}});
    auto static_beam = create_test_static_beam_parameters();

    auto tangent_operator = Kokkos::View<double[6][6]>("tangent_operator");
    static_beam.TangentOperator(psi, 1., tangent_operator);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        tangent_operator,
        {
            {1., 0., 0., 0., 0., 0.},                                                      // row 1
            {0., 1., 0., 0., 0., 0.},                                                      // row 2
            {0., 0., 1., 0., 0., 0.},                                                      // row 3
            {0., 0., 0., -0.06871266098996709, 0.555552845761836, -0.014131010177901665},  // row 4
            {0., 0., 0., -0.2267181808418461, 0.1779133377000255, 0.6236305018139318},     // row 5
            {0., 0., 0., 0.5073830075578865, 0.36287349294603777, 0.5889566688500127}      // row 6
        }
    );
}

TEST(StaticBeamTest, ConstraintsGradientMatrix) {
    auto constraint_gradients = Kokkos::View<double[6][30]>("constraint_gradients");
    BMatrix(constraint_gradients);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        constraint_gradients,
        {
            {1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},  // row 3
            {0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},  // row 4
            {0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},  // row 5
            {0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}  // row 6
        }
    );
}

struct PopulatePositionVectors {
    Kokkos::View<double[5][7]> position_vectors;
    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        position_vectors(0, 0) = 1.;
        position_vectors(0, 1) = 0.;
        position_vectors(0, 2) = 0.;
        position_vectors(0, 3) = 1.;
        position_vectors(0, 4) = 0.;
        position_vectors(0, 5) = 0.;
        position_vectors(0, 6) = 0.;
        // node 2
        position_vectors(1, 0) = 2.7267316464601146;
        position_vectors(1, 1) = 0.;
        position_vectors(1, 2) = 0.;
        position_vectors(1, 3) = 1.;
        position_vectors(1, 4) = 0.;
        position_vectors(1, 5) = 0.;
        position_vectors(1, 6) = 0.;
        // node 3
        position_vectors(2, 0) = 6.;
        position_vectors(2, 1) = 0.;
        position_vectors(2, 2) = 0.;
        position_vectors(2, 3) = 1.;
        position_vectors(2, 4) = 0.;
        position_vectors(2, 5) = 0.;
        position_vectors(2, 6) = 0.;
        // node 4
        position_vectors(3, 0) = 9.273268353539885;
        position_vectors(3, 1) = 0.;
        position_vectors(3, 2) = 0.;
        position_vectors(3, 3) = 1.;
        position_vectors(3, 4) = 0.;
        position_vectors(3, 5) = 0.;
        position_vectors(3, 6) = 0.;
        // node 5
        position_vectors(4, 0) = 11.;
        position_vectors(4, 1) = 0.;
        position_vectors(4, 2) = 0.;
        position_vectors(4, 3) = 1.;
        position_vectors(4, 4) = 0.;
        position_vectors(4, 5) = 0.;
        position_vectors(4, 6) = 0.;
    }
};

TEST(StaticBeamTest, StaticBeamResidual) {
    auto position_vectors = Kokkos::View<double[5][7]>("position_vectors");
    Kokkos::parallel_for(1, PopulatePositionVectors{position_vectors});

    // Stiffness matrix for uniform composite beam section (in material csys)
    auto stiffness = gen_alpha_solver::create_matrix({
        {1.36817e6, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });

    // Use a 7-point Gauss-Legendre quadrature for integration
    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    StaticBeamLinearizationParameters static_beam{
        position_vectors, StiffnessMatrix(stiffness), quadrature};

    auto gen_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},    // node 1
        {0., 0., 0., 1., 0., 0., 0.},    // node 2
        {0., 0., 0., 1., 0., 0., 0.},    // node 3
        {0., 0., 0., 1., 0., 0., 0.},    // node 4
        {0., 0.001, 0., 1., 0., 0., 0.}  // node 5
    });

    auto velocity = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.}}
    );

    auto acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.}}
    );

    auto lagrange_mults = gen_alpha_solver::create_vector({0., 0., 0., 0., 0., 0.});

    auto residual = Kokkos::View<double[36]>("residual");
    static_beam.ResidualVector(
        gen_coords, velocity, acceleration, lagrange_mults, gen_alpha_solver::TimeStepper(), residual
    );

    std::vector<double> expected = {0., 0.8856000000000164,
                                    0., 0.,
                                    0., 4.428,
                                    0., -3.018979543801465,
                                    0., 0.,
                                    0., -12.488413959858132,
                                    0., 9.446400000000166,
                                    0., 0.,
                                    0., 23.61600000000001,
                                    0., -69.30502045619781,
                                    0., 0.,
                                    0., -59.83558604014186,
                                    0., 61.99199999999924,
                                    0., 0.,
                                    0., -44.28000000000008,
                                    0., 0.,
                                    0., 0.,
                                    0., 0.};
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(residual, expected);
}

TEST(StaticBeamTest, StaticBeamIterationMatrix) {
    auto position_vectors = Kokkos::View<double[5][7]>("position_vectors");
    Kokkos::parallel_for(1, PopulatePositionVectors{position_vectors});

    // Stiffness matrix for uniform composite beam section (in material csys)
    auto stiffness = gen_alpha_solver::create_matrix({
        {1.36817e6, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });

    // Use a 7-point Gauss-Legendre quadrature for integration
    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    StaticBeamLinearizationParameters static_beam{
        position_vectors, StiffnessMatrix(stiffness), quadrature};

    auto gen_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},    // node 1
        {0., 0., 0., 1., 0., 0., 0.},    // node 2
        {0., 0., 0., 1., 0., 0., 0.},    // node 3
        {0., 0., 0., 1., 0., 0., 0.},    // node 4
        {0., 0.001, 0., 1., 0., 0., 0.}  // node 5
    });

    auto delta_gen_coords = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.}}
    );

    auto velocity = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.}}
    );

    auto acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.}}
    );

    auto lagrange_mults = gen_alpha_solver::create_vector({0., 0., 0., 0., 0., 0.});

    auto h = 1.;
    auto beta_prime = 0.25;
    auto gamma_prime = 0.5;

    auto iteration_matrix = Kokkos::View<double[36][36]>("iteration_matrix");
    static_beam.IterationMatrix(
        h, beta_prime, gamma_prime, gen_coords, delta_gen_coords, velocity, acceleration,
        lagrange_mults, iteration_matrix
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        iteration_matrix, expected_iteration
    );
}

TEST(StaticCompositeBeamTest, StaticAnalysisWithZeroForceAndNonZeroInitialGuess) {
    auto position_vectors = Kokkos::View<double[5][7]>("position_vectors");
    Kokkos::parallel_for(1, PopulatePositionVectors{position_vectors});

    // Stiffness matrix for uniform composite beam section (in material csys)
    auto stiffness = gen_alpha_solver::create_matrix({
        {1.36817e6, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });

    // Use a 7-point Gauss-Legendre quadrature for integration
    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto gen_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},    // node 1
        {0., 0., 0., 1., 0., 0., 0.},    // node 2
        {0., 0., 0., 1., 0., 0., 0.},    // node 3
        {0., 0., 0., 1., 0., 0., 0.},    // node 4
        {0., 0.001, 0., 1., 0., 0., 0.}  // node 5
    });

    auto v = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );

    auto velocity = v;
    auto acceleration = v;
    auto algo_acceleration = v;
    auto initial_state = State(gen_coords, velocity, acceleration, algo_acceleration);

    auto lagrange_mults = gen_alpha_solver::create_vector({0., 0., 0., 0., 0., 0.});
    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        0., 0., 0.5, 1., gen_alpha_solver::TimeStepper(0., 1., 1, 20), false
    );
    std::shared_ptr<LinearizationParameters> static_beam_lin_params =
        std::make_shared<StaticBeamLinearizationParameters>(
            position_vectors, StiffnessMatrix(stiffness), quadrature
        );
    auto results =
        time_integrator.Integrate(initial_state, lagrange_mults.extent(0), static_beam_lin_params);
    auto final_state = results.back();

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.GetGeneralizedCoordinates(),
        {
            {0., 0., 0., 1., 0., 0., 0.},  // node 1
            {0., 0., 0., 1., 0., 0., 0.},  // node 2
            {0., 0., 0., 1., 0., 0., 0.},  // node 3
            {0., 0., 0., 1., 0., 0., 0.},  // node 4
            {0., 0., 0., 1., 0., 0., 0.}   // node 5
        }
    );
}

}  // namespace openturbine::gebt_poc::tests
