#include <gtest/gtest.h>

#include "src/gebt_poc/state.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(StateTest, CreateStateWithGivenValues) {
    auto state = State{
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
        })};

    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.generalized_coordinates,
        {
            {1., 2., 3., 4., 0., 0., 0.}  // 7 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.velocity,
        {
            {5., 6., 7., 0., 0., 0.}  // 6 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.acceleration,
        {
            {5., 6., 7., 0., 0., 0.}  // 6 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        state.algorithmic_acceleration,
        {
            {5., 6., 7., 0., 0., 0.}  // 6 elements
        }
    );
}

}  // namespace openturbine::gebt_poc::tests
