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

TEST(TimeIntegratorTest, AdvanceAnalysisTimeByNumberofSteps) {
    auto state = State();
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 1.0, 10, state);

    EXPECT_EQ(time_integrator.GetCurrentTime(), 0.);

    time_integrator.Integrate();

    EXPECT_EQ(time_integrator.GetCurrentTime(), 10.0);
}

TEST(StateTest, CreateDefaultState) {
    auto state = State();

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {0.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedVelocity(), {0.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedAcceleration(), {0.});
    expect_kokkos_view_1D_equal(state.GetAccelerations(), {0.});
}

TEST(StateTest, CreateState) {
    auto v = create_vector({1., 2., 3.});
    auto state = State(v, v, v, v);

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedVelocity(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(state.GetGeneralizedAcceleration(), {1., 2., 3.});
    expect_kokkos_view_1D_equal(state.GetAccelerations(), {1., 2., 3.});
}

TEST(TimeIntegratorTest, GetCurrentStateFromTimeIntegrator) {
    auto state = State();
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0., 0.10, 10, state);

    EXPECT_EQ(time_integrator.GetCurrentTime(), 0.);

    time_integrator.Integrate();
    auto current_state = time_integrator.GetState();

    EXPECT_NEAR(time_integrator.GetCurrentTime(), 1., std::numeric_limits<double>::epsilon());
}

}  // namespace openturbine::rigid_pendulum::tests
