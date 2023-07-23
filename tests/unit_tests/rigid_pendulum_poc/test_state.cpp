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

TEST(MassMatrixTest, CreateMassMatrixWithDefaultValues) {
    auto mass_matrix = MassMatrix();
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
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

TEST(MassMatrixTest, CreateMassMatrixWithGivenValues) {
    auto mass_matrix = MassMatrix(1., 2.);
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0.},  // row 3
            {0., 0., 0., 2., 0., 0.},  // row 4
            {0., 0., 0., 0., 2., 0.},  // row 5
            {0., 0., 0., 0., 0., 2.}   // row 6
        }
    );
}

TEST(MassMatrixTest, ExpectMassMatrixToThrowWhenMassIsZero) {
    EXPECT_THROW(MassMatrix(0., 1.), std::invalid_argument);
}

TEST(MassMatrixTest, ExpectMassMatrixToThrowWhenMomentOfInertiaIsZero) {
    EXPECT_THROW(MassMatrix(1., 0.), std::invalid_argument);
}

TEST(MassMatrixTest, CreateMassMatrixWithGiven2DVector) {
    auto mass_matrix = MassMatrix(create_matrix({
        {1., 2., 3., 4., 5., 6.},        // row 1
        {7., 8., 9., 10., 11., 12.},     // row 2
        {13., 14., 15., 16., 17., 18.},  // row 3
        {19., 20., 21., 22., 23., 24.},  // row 4
        {25., 26., 27., 28., 29., 30.},  // row 5
        {31., 32., 33., 34., 35., 36.}   // row 6
    }));
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {1., 2., 3., 4., 5., 6.},        // row 1
            {7., 8., 9., 10., 11., 12.},     // row 2
            {13., 14., 15., 16., 17., 18.},  // row 3
            {19., 20., 21., 22., 23., 24.},  // row 4
            {25., 26., 27., 28., 29., 30.},  // row 5
            {31., 32., 33., 34., 35., 36.}   // row 6
        }
    );
}

}  // namespace openturbine::rigid_pendulum::tests
