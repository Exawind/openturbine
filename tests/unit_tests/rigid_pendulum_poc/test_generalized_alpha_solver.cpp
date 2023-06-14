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

TEST(StateTest, AddTwoStatesWithAdditionOperator) {
    auto v = create_vector({1., 2., 3.});
    auto state1 = State(v, v, v, v);
    auto state2 = State(v, v, v, v);

    auto state3 = state1 + state2;

    expect_kokkos_view_1D_equal(state3.GetGeneralizedCoordinates(), {2., 4., 6.});
    expect_kokkos_view_1D_equal(state3.GetGeneralizedVelocity(), {2., 4., 6.});
    expect_kokkos_view_1D_equal(state3.GetGeneralizedAcceleration(), {2., 4., 6.});
    expect_kokkos_view_1D_equal(state3.GetAlgorithmicAcceleration(), {2., 4., 6.});
}

TEST(StateTest, AddTwoStatesWithAdditionAssignmentOperator) {
    auto v1 = create_vector({0.42, -1.17, 3.14});
    auto state1 = State(v1, v1, v1, v1);
    auto v2 = create_vector({2.31, 0.93, -71.26});
    auto state2 = State(v2, v2, v2, v2);

    state1 += state2;

    expect_kokkos_view_1D_equal(state1.GetGeneralizedCoordinates(), {2.73, -0.24, -68.12});
    expect_kokkos_view_1D_equal(state1.GetGeneralizedVelocity(), {2.73, -0.24, -68.12});
    expect_kokkos_view_1D_equal(state1.GetGeneralizedAcceleration(), {2.73, -0.24, -68.12});
    expect_kokkos_view_1D_equal(state1.GetAlgorithmicAcceleration(), {2.73, -0.24, -68.12});
}

}  // namespace openturbine::rigid_pendulum::tests
