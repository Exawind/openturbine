#include <limits>

#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/generalized_alpha_time_integrator.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(TimeIntegratorTest, GetTimeIntegratorType) {
    auto time_integrator =
        GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.25, 0.5, TimeStepper(0., 1.0, 10));

    EXPECT_EQ(time_integrator.GetType(), TimeIntegratorType::GENERALIZED_ALPHA);
}

// TEST(TimeIntegratorTest, AdvanceAnalysisTimeByNumberOfSteps) {
//     auto time_integrator =
//         GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.25, 0.5, TimeStepper(0., 1.0, 10));

//     EXPECT_EQ(time_integrator.GetTimeStepper().GetCurrentTime(), 0.);

//     auto initial_state = State();
//     time_integrator.Integrate(initial_state);

//     EXPECT_EQ(time_integrator.GetTimeStepper().GetCurrentTime(), 10.0);
// }

// TEST(TimeIntegratorTest, GetHistoryOfStatesFromTimeIntegrator) {
//     auto time_integrator =
//         GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.25, 0.5, TimeStepper(0., 0.1, 17));

//     EXPECT_EQ(time_integrator.GetTimeStepper().GetCurrentTime(), 0.);

//     auto initial_state = State();
//     auto state_history = time_integrator.Integrate(initial_state);

//     EXPECT_NEAR(
//         time_integrator.GetTimeStepper().GetCurrentTime(), 1.70,
//         10 * std::numeric_limits<double>::epsilon()
//     );
//     EXPECT_EQ(state_history.size(), 18);
// }

// TEST(TimeIntegratorTest, LinearSolutionWithZeroAcceleration) {
//     auto initial_state = State();
//     auto time_integrator =
//         GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.25, 0.5, TimeStepper(0., 1., 1));
//     auto linear_update = time_integrator.UpdateLinearSolution(initial_state);

//     expect_kokkos_view_1D_equal(linear_update.GetGeneralizedCoordinates(), {0.});
//     expect_kokkos_view_1D_equal(linear_update.GetGeneralizedVelocity(), {0.});
//     expect_kokkos_view_1D_equal(linear_update.GetGeneralizedAcceleration(), {0.});
// }

// TEST(TimeIntegratorTest, LinearSolutionWithNonZeroAcceleration) {
//     auto v = create_vector({1., 2., 3.});
//     auto initial_state = State(v, v, v, v);
//     auto time_integrator =
//         GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.25, 0.5, TimeStepper(0., 1., 1));
//     auto linear_update = time_integrator.UpdateLinearSolution(initial_state);

//     expect_kokkos_view_1D_equal(linear_update.GetGeneralizedCoordinates(), {2.25, 4.5, 6.75});
//     expect_kokkos_view_1D_equal(linear_update.GetGeneralizedVelocity(), {1.5, 3., 4.5});
//     expect_kokkos_view_1D_equal(linear_update.GetGeneralizedAcceleration(), {0., 0., 0.});
// }

// TEST(TimeIntegratorTest, TotalNumberOfIterationsInNonLinearSolution) {
//     auto time_integrator =
//         GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.25, 0.5, TimeStepper(0., 1., 10));

//     EXPECT_EQ(time_integrator.GetTimeStepper().GetNumberOfIterations(), 0);
//     EXPECT_EQ(time_integrator.GetTimeStepper().GetTotalNumberOfIterations(), 0);

//     auto initial_state = State();
//     time_integrator.Integrate(initial_state);

//     EXPECT_LE(
//         time_integrator.GetTimeStepper().GetNumberOfIterations(),
//         time_integrator.GetTimeStepper().GetMaximumNumberOfIterations()
//     );
//     EXPECT_LE(
//         time_integrator.GetTimeStepper().GetTotalNumberOfIterations(),
//         time_integrator.GetTimeStepper().GetNumberOfSteps() *
//             time_integrator.GetTimeStepper().GetMaximumNumberOfIterations()
//     );
// }

// TEST(TimeIntegratorTest, ExpectConvergedSolution) {
//     auto residual_force = create_vector({1.e-7, 2.e-7, 3.e-7});
//     auto incremental_force = create_vector({1., 2., 3.});
//     auto time_integrator = GeneralizedAlphaTimeIntegrator();
//     auto converged = time_integrator.CheckConvergence(residual_force, incremental_force);

//     EXPECT_TRUE(converged);
// }

// TEST(TimeIntegratorTest, ExpectNonConvergedSolution) {
//     auto residual_force = create_vector({1.e-3, 2.e-3, 3.e-3});
//     auto incremental_force = create_vector({1., 2., 3.});
//     auto time_integrator = GeneralizedAlphaTimeIntegrator();
//     auto converged = time_integrator.CheckConvergence(residual_force, incremental_force);

//     EXPECT_FALSE(converged);
// }

TEST(GeneralizedAlphaTimeIntegratorTest, ConstructorWithInvalidAlphaF) {
    EXPECT_THROW(GeneralizedAlphaTimeIntegrator(1.1, 0.5, 0.25, 0.5), std::invalid_argument);
}

TEST(GeneralizedAlphaTimeIntegratorTest, ConstructorWithInvalidAlphaM) {
    EXPECT_THROW(GeneralizedAlphaTimeIntegrator(0.5, 1.1, 0.25, 0.5), std::invalid_argument);
}

TEST(GeneralizedAlphaTimeIntegratorTest, ConstructorWithInvalidBeta) {
    EXPECT_THROW(GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.75, 0.5), std::invalid_argument);
}

TEST(GeneralizedAlphaTimeIntegratorTest, ConstructorWithInvalidGamma) {
    EXPECT_THROW(GeneralizedAlphaTimeIntegrator(0.5, 0.5, 0.25, 1.1), std::invalid_argument);
}

TEST(GeneralizedAlphaTimeIntegratorTest, GetDefaultGAConstants) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator();

    EXPECT_EQ(time_integrator.GetAlphaF(), 0.5);
    EXPECT_EQ(time_integrator.GetAlphaM(), 0.5);
    EXPECT_EQ(time_integrator.GetBeta(), 0.25);
    EXPECT_EQ(time_integrator.GetGamma(), 0.5);
}

TEST(GeneralizedAlphaTimeIntegratorTest, GetSuppliedGAConstants) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0.11, 0.29, 0.47, 0.93);

    EXPECT_EQ(time_integrator.GetAlphaF(), 0.11);
    EXPECT_EQ(time_integrator.GetAlphaM(), 0.29);
    EXPECT_EQ(time_integrator.GetBeta(), 0.47);
    EXPECT_EQ(time_integrator.GetGamma(), 0.93);
}

// TEST(TimeIntegratorTest, AlphaStepSolutionAfterOneIncWithZeroAcceleration) {
//     auto initial_state = State();
//     auto time_integrator =
//         GeneralizedAlphaTimeIntegrator(0., 0., 0.5, 1., TimeStepper(0., 1., 1, 1));

//     EXPECT_EQ(time_integrator.GetTimeStepper().GetNumberOfIterations(), 0);
//     EXPECT_EQ(time_integrator.GetTimeStepper().GetTotalNumberOfIterations(), 0);

//     auto results = time_integrator.Integrate(initial_state);

//     EXPECT_EQ(time_integrator.GetTimeStepper().GetNumberOfIterations(), 1);
//     EXPECT_EQ(time_integrator.GetTimeStepper().GetTotalNumberOfIterations(), 1);

//     auto final_state = results.back();

//     // We expect the final state to contain the following values after one increment
//     expect_kokkos_view_1D_equal(final_state.GetGeneralizedCoordinates(), {1.});
//     expect_kokkos_view_1D_equal(final_state.GetGeneralizedVelocity(), {2.});
//     expect_kokkos_view_1D_equal(final_state.GetGeneralizedAcceleration(), {2.});
//     expect_kokkos_view_1D_equal(final_state.GetAlgorithmicAcceleration(), {2.});
// }

// TEST(TimeIntegratorTest, AlphaStepSolutionAfterTwoIncsWithZeroAcceleration) {
//     auto initial_state = State();
//     auto time_integrator =
//         GeneralizedAlphaTimeIntegrator(0., 0., 0.5, 1., TimeStepper(0., 1., 1, 2));
//     auto results = time_integrator.Integrate(initial_state);

//     EXPECT_EQ(time_integrator.GetTimeStepper().GetNumberOfIterations(), 2);
//     EXPECT_EQ(time_integrator.GetTimeStepper().GetTotalNumberOfIterations(), 2);

//     auto final_state = results.back();

//     // We expect the final state to contain the following values after two increments
//     expect_kokkos_view_1D_equal(final_state.GetGeneralizedCoordinates(), {2.});
//     expect_kokkos_view_1D_equal(final_state.GetGeneralizedVelocity(), {4.});
//     expect_kokkos_view_1D_equal(final_state.GetGeneralizedAcceleration(), {4.});
//     expect_kokkos_view_1D_equal(final_state.GetAlgorithmicAcceleration(), {4.});
// }

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateTangentDampingMatrix) {
    auto angular_velocity_vector = create_vector({0.3, 0.1, 0.8});

    auto mass = 15.;
    auto mass_matrix = MassMatrix(mass, Vector(0.234375, 0.46875, 0.234375));
    auto inertia_matrix = mass_matrix.GetMomentOfInertiaMatrix();

    auto tangent_damping_matrix =
        heavy_top_tangent_damping_matrix(angular_velocity_vector, inertia_matrix);

    expect_kokkos_view_2D_equal(
        tangent_damping_matrix,
        {
            {0., 0., 0., 0., 0., 0.},               // row 1
            {0., 0., 0., 0., 0., 0.},               // row 2
            {0., 0., 0., 0., 0., 0.},               // row 3
            {0., 0., 0., 0., -0.1875, -0.0234375},  // row 4
            {0., 0., 0., 0., 0., 0.},               // row 5
            {0., 0., 0., 0.0234375, 0.0703125, 0.}  // row 6
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateTangentStiffnessMatrix) {
    auto position_vector = create_vector({0., 1., 0.});
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );
    auto lagrange_multipliers = create_vector({1., 2., 3.});

    auto tangent_stiffness_matrix =
        heavy_top_tangent_stiffness_matrix(position_vector, rotation_matrix, lagrange_multipliers);

    expect_kokkos_view_2D_equal(
        tangent_stiffness_matrix,
        {
            {0., 0., 0., 0., 0., 0.},               // row 1
            {0., 0., 0., 0., 0., 0.},               // row 2
            {0., 0., 0., 0., 0., 0.},               // row 3
            {0., 0., 0., -1.002378, 1.779458, 0.},  // row 4
            {0., 0., 0., 0., 0., 0.},               // row 5
            {0., 0., 0., 0., 3.135086, -1.002378}   // row 6
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateConstraintGradientMatrix) {
    auto position_vector = create_vector({0., 1., 0.});
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );

    auto constraint_gradient_matrix =
        heavy_top_constraint_gradient_matrix(position_vector, rotation_matrix);

    expect_kokkos_view_2D_equal(
        constraint_gradient_matrix,
        {
            {-1., 0., 0., 0.210962, 0., -0.617251},     // row 1
            {0., -1., 0., -0.00416521, 0., -0.775967},  // row 2
            {0., 0., -1., 0.977485, 0., 0.129909}       // row 3
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateIterationMatrix) {
    auto M = MassMatrix(15., Vector(0.234375, 0.46875, 0.234375));
    auto mass_matrix = M.GetMassMatrix();
    auto inertia_matrix = M.GetMomentOfInertiaMatrix();
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );
    auto angular_velocity_vector = create_vector({0.3, 0.1, 0.8});
    auto position_vector = create_vector({0., 1., 0.});
    auto lagrange_multipliers = create_vector({1., 2., 3.});
    const auto BETA_PRIME = 1.;
    const auto GAMMA_PRIME = 2.;

    auto iteration_matrix = heavy_top_iteration_matrix(
        mass_matrix, inertia_matrix, rotation_matrix, angular_velocity_vector, position_vector,
        lagrange_multipliers, BETA_PRIME, GAMMA_PRIME
    );

    expect_kokkos_view_2D_equal(
        iteration_matrix,
        {
            {15., 0., 0., 0., 0., 0., -1., 0., 0.},                                       // row 1
            {0., 15., 0., 0., 0., 0., 0., -1., 0.},                                       // row 2
            {0., 0., 15., 0., 0., 0., 0., 0., -1.},                                       // row 3
            {0., 0., 0., -0.768003, 1.404458, -0.046875, 0.210962, -0.004165, 0.977485},  // row 4
            {0., 0., 0., 0., 0.46875, 0., 0., 0., 0.},                                    // row 5
            {0., 0., 0., 0.046875, 3.275712, -0.768003, -0.617251, -0.775967, 0.129909},  // row 6
            {-1., 0., 0., 0.210962, 0., -0.617251, 0., 0., 0.},                           // row 7
            {0., -1., 0., -0.004165, 0., -0.775967, 0., 0., 0.},                          // row 8
            {0., 0., -1., 0.977485, 0., 0.129909, 0., 0., 0.}                             // row 9
        }
    );
}

}  // namespace openturbine::rigid_pendulum::tests
