#include <gtest/gtest.h>

#include "src/gebt_poc/state.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(StateTest, CreateDefaultState) {
    auto state = State();

    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetGeneralizedCoordinates(),
        {
            {0., 0., 0., 0., 0., 0., 0.}  // 7 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetVelocity(),
        {
            {0., 0., 0., 0., 0., 0.}  // 6 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetAcceleration(),
        {
            {0., 0., 0., 0., 0., 0.}  // 6 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetAlgorithmicAcceleration(),
        {
            {0., 0., 0., 0., 0., 0.}  // 6 elements
        }
    );
}

TEST(StateTest, CreateStateWithGivenValues) {
    auto state = State(
        gen_alpha_solver::create_matrix({
            {1., 2., 3., 4., 0., 0., 0.},  // 7 elements
        }),
        gen_alpha_solver::create_matrix({
            {5., 6., 7., 0., 0., 0.},  // 6 elements
        }),
        gen_alpha_solver::create_matrix({
            {5., 6., 7., 0., 0., 0.},  // 6 elements
        }),
        gen_alpha_solver::create_matrix({
            {5., 6., 7., 0., 0., 0.},  // 6 elements
        })
    );

    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetGeneralizedCoordinates(),
        {
            {1., 2., 3., 4., 0., 0., 0.}  // 7 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetVelocity(),
        {
            {5., 6., 7., 0., 0., 0.}  // 6 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetAcceleration(),
        {
            {5., 6., 7., 0., 0., 0.}  // 6 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.GetAlgorithmicAcceleration(),
        {
            {5., 6., 7., 0., 0., 0.}  // 6 elements
        }
    );
}

TEST(StateTest, ExpectFailureIfNumberOfInputsDoNotMatchInGeneralizedCoordinates) {
    EXPECT_THROW(
        State(
            gen_alpha_solver::create_matrix({
                {1., 1., 1., 1., 1., 1., 1.},  // row 1
                {1., 1., 1., 1., 1., 1., 1.}   // row 2
            }),
            gen_alpha_solver::create_matrix({
                {2., 2., 2., 2., 2., 2.}  // row 1
            }),
            gen_alpha_solver::create_matrix({
                {3., 3., 3., 3., 3., 3.}  // row 1
            }),
            gen_alpha_solver::create_matrix({
                {4., 4., 4., 4., 4., 4.}  // row 1
            })
        ),
        std::invalid_argument
    );
}

TEST(StateTest, ExpectFailureIfNumberOfInputsDoNotMatchInAcceleration) {
    EXPECT_THROW(
        State(
            gen_alpha_solver::create_matrix({
                {1., 1., 1., 1., 1., 1., 1.},  // row 1
            }),
            gen_alpha_solver::create_matrix({
                {2., 2., 2., 2., 2., 2.}  // row 1
            }),
            gen_alpha_solver::create_matrix({
                {3., 3., 3., 3., 3., 3.},  // row 1
                {4., 4., 4., 4., 4., 4.}   // row 2
            }),
            gen_alpha_solver::create_matrix({
                {5., 5., 5., 5., 5., 5.}  // row 1
            })
        ),
        std::invalid_argument
    );
}

}  // namespace openturbine::gebt_poc::tests
