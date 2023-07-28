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

// bool GeneralizedAlphaTimeIntegrator::CheckConvergence(HostView1D residual) {
//     // L2 norm of the residual vector should be very small (< epsilon) for the solution
//     // to be considered converged
//     double residual_norm = 0.;
//     Kokkos::parallel_reduce(
//         residual.extent(0),
//         KOKKOS_LAMBDA(int i, double& residual_partial_sum) {
//             double residual_value = residual(i);
//             residual_partial_sum += residual_value * residual_value;
//         },
//         Kokkos::Sum<double>(residual_norm)
//     );
//     residual_norm = std::sqrt(residual_norm);

//     auto log = util::Log::Get();
//     log->Debug(
//         "Residual norm: " + std::to_string(residual_norm) + "\n"
//     );

//     return (residual_norm) < kTOLERANCE ? true : false;
// }

TEST(TimeIntegratorTest, ExpectConvergedSolution) {
    auto residual = create_vector({1.e-7, 2.e-7, 3.e-7});
    auto time_integrator = GeneralizedAlphaTimeIntegrator();
    auto converged = time_integrator.CheckConvergence(residual);

    EXPECT_TRUE(converged);
}

TEST(TimeIntegratorTest, ExpectNonConvergedSolution) {
    auto residual = create_vector({1.e-5, 2.e-5, 3.e-5});
    auto time_integrator = GeneralizedAlphaTimeIntegrator();
    auto converged = time_integrator.CheckConvergence(residual);

    EXPECT_FALSE(converged);
}

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

}  // namespace openturbine::rigid_pendulum::tests
