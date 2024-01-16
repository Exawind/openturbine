#include "tests/unit_tests/gebt_poc/test_johnson_static_beam.h"

#include <gtest/gtest.h>

#include "src/gebt_poc/gen_alpha_2D.h"
#include "src/gebt_poc/solver.h"
#include "src/gebt_poc/static_beam_element.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(StaticBeamTest, CalculateTangentOperatorWithPhiAsZero) {
    auto psi = gen_alpha_solver::create_vector({0., 0., 0.});
    StaticBeamLinearizationParameters static_beam{};

    auto tangent_operator = Kokkos::View<double[6][6]>("tangent_operator");
    static_beam.TangentOperator(psi, tangent_operator);

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
    auto psi = gen_alpha_solver::create_vector({1., 2., 3.});
    StaticBeamLinearizationParameters static_beam{};

    auto tangent_operator = Kokkos::View<double[6][6]>("tangent_operator");
    static_beam.TangentOperator(psi, tangent_operator);

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

    StaticBeamLinearizationParameters static_beam{position_vectors, stiffness, quadrature};

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

    StaticBeamLinearizationParameters static_beam{position_vectors, stiffness, quadrature};

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
        std::make_shared<StaticBeamLinearizationParameters>(position_vectors, stiffness, quadrature);
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
