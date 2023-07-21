#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/state.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(StateTest, CreateDefaultState) {
    auto state = State();

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {0.});
    expect_kokkos_view_1D_equal(state.GetVelocity(), {0.});
    expect_kokkos_view_1D_equal(state.GetAcceleration(), {0.});
    expect_kokkos_view_1D_equal(state.GetAlgorithmicAcceleration(), {0.});
}

TEST(StateTest, CreateStateWithGivenValues) {
    auto q = create_vector({1., 2., 3., 4.});
    auto v = create_vector({5., 6., 7.});
    auto state = State(q, v, v, v);

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {1., 2., 3., 4.});
    expect_kokkos_view_1D_equal(state.GetVelocity(), {5., 6., 7.});
    expect_kokkos_view_1D_equal(state.GetAcceleration(), {5., 6., 7.});
    expect_kokkos_view_1D_equal(state.GetAlgorithmicAcceleration(), {5., 6., 7.});
}

}  // namespace openturbine::rigid_pendulum::tests
