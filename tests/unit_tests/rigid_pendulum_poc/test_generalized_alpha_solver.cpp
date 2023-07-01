#include <limits>

#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/solver.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(TimeIntegratorTest, CreateDefaultTimeIntegrator) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator();

    EXPECT_EQ(time_integrator.GetInitialTime(), 0.);
    EXPECT_EQ(time_integrator.GetCurrentTime(), 0.);
    EXPECT_EQ(time_integrator.GetTimeStep(), 1.);
    EXPECT_EQ(time_integrator.GetNumberOfSteps(), 1);
}

TEST(TimeIntegratorTest, CreateTimeIntegrator) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(1., 0.01, 10);

    EXPECT_EQ(time_integrator.GetInitialTime(), 1.);
    EXPECT_EQ(time_integrator.GetCurrentTime(), 1.);
    EXPECT_EQ(time_integrator.GetTimeStep(), 0.01);
    EXPECT_EQ(time_integrator.GetNumberOfSteps(), 10);
}

TEST(TimeIntegratorTest, AdvanceAnalysisTime) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator();

    EXPECT_EQ(time_integrator.GetCurrentTime(), 0.);

    time_integrator.AdvanceTimeStep();
    EXPECT_EQ(time_integrator.GetCurrentTime(), 1.);
}

TEST(StateTest, CreateDefaultState) {
    auto state = State();

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {0.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedVelocity(), {0.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedAcceleration(), {0.});
    expect_kokkos_view_1D_equal(state.GetAlgorithmicAcceleration(), {0.});
}

TEST(StateTest, CreateState) {
    auto v = create_vector({1., 2., 3.});
    auto state = State(v, v, v, v);

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedVelocity(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedAcceleration(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(state.GetAlgorithmicAcceleration(), {1., 2., 3.});
}

TEST(TimeIntegratorTest, AdvanceAnalysisTimeByNumberOfSteps) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 1.0, 10);

    EXPECT_EQ(time_integrator.GetCurrentTime(), 0.);

    auto initial_state = State();
    time_integrator.Integrate(initial_state);

    EXPECT_EQ(time_integrator.GetCurrentTime(), 10.0);
}

TEST(TimeIntegratorTest, GetHistoryOfStatesFromTimeIntegrator) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 0.10, 17);

    EXPECT_EQ(time_integrator.GetCurrentTime(), 0.);

    auto initial_state = State();
    auto state_history = time_integrator.Integrate(initial_state);

    EXPECT_NEAR(time_integrator.GetCurrentTime(), 1.70, 10 * std::numeric_limits<double>::epsilon());
    EXPECT_EQ(state_history.size(), 18);
}

TEST(TimeIntegratorTest, LinearSolutionWithZeroAcceleration) {
    auto initial_state = State();

    expect_kokkos_view_1D_equal(initial_state.GetGeneralizedCoordinates(), {0.});
    expect_kokkos_view_1D_equal(initial_state.GetGeneralizedVelocity(), {0.});
    expect_kokkos_view_1D_equal(initial_state.GetGeneralizedAcceleration(), {0.});
    expect_kokkos_view_1D_equal(initial_state.GetAlgorithmicAcceleration(), {0.});

    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 1., 1);
    auto [linear_coords, linear_velocity, algo_acceleration] = time_integrator.UpdateLinearSolution(
        initial_state.GetGeneralizedCoordinates(), initial_state.GetGeneralizedVelocity(),
        initial_state.GetGeneralizedAcceleration(), initial_state.GetAlgorithmicAcceleration()
    );

    expect_kokkos_view_1D_equal(linear_coords, {0.});
    expect_kokkos_view_1D_equal(linear_velocity, {0.});
    expect_kokkos_view_1D_equal(algo_acceleration, {0.});
}

TEST(TimeIntegratorTest, LinearSolutionWithNonZeroAcceleration) {
    auto v = create_vector({1., 2., 3.});
    auto initial_state = State(v, v, v, v);

    expect_kokkos_view_1D_equal(initial_state.GetGeneralizedCoordinates(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(initial_state.GetGeneralizedVelocity(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(initial_state.GetGeneralizedAcceleration(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(initial_state.GetAlgorithmicAcceleration(), {1., 2., 3.});

    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 1., 1);
    auto [linear_coords, linear_velocity, algo_acceleration] = time_integrator.UpdateLinearSolution(
        initial_state.GetGeneralizedCoordinates(), initial_state.GetGeneralizedVelocity(),
        initial_state.GetGeneralizedAcceleration(), initial_state.GetAlgorithmicAcceleration()
    );

    expect_kokkos_view_1D_equal(linear_coords, {2.25, 4.5, 6.75});
    expect_kokkos_view_1D_equal(linear_velocity, {1.5, 3., 4.5});
    expect_kokkos_view_1D_equal(algo_acceleration, {0., 0., 0.});
}

TEST(TimeIntegratorTest, TotalNumberOfIterationsInNonLinearSolution) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 1.0, 10);

    EXPECT_EQ(time_integrator.GetNumberOfIterations(), 0);
    EXPECT_EQ(time_integrator.GetTotalNumberOfIterations(), 0);

    auto initial_state = State();
    time_integrator.Integrate(initial_state);

    EXPECT_LE(time_integrator.GetNumberOfIterations(), time_integrator.kMAX_ITERATIONS);
    EXPECT_LE(
        time_integrator.GetTotalNumberOfIterations(),
        time_integrator.GetNumberOfSteps() * time_integrator.kMAX_ITERATIONS
    );
}

TEST(TimeIntegratorTest, ExpectConvergedSolution) {
    auto residual_force = create_vector({1.e-7, 2.e-7, 3.e-7});
    auto incremental_force = create_vector({1., 2., 3.});
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 1., 1);
    auto converged = time_integrator.CheckConvergence(residual_force, incremental_force);

    EXPECT_TRUE(converged);
}

TEST(TimeIntegratorTest, ExpectNonConvergedSolution) {
    auto residual_force = create_vector({1.e-3, 2.e-3, 3.e-3});
    auto incremental_force = create_vector({1., 2., 3.});
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 1., 1);
    auto converged = time_integrator.CheckConvergence(residual_force, incremental_force);

    EXPECT_FALSE(converged);
}

}  // namespace openturbine::rigid_pendulum::tests
