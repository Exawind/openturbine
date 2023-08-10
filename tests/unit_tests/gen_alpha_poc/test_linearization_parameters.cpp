#include <gtest/gtest.h>

#include "src/gen_alpha_poc/linearization_parameters.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gen_alpha_solver::tests {

TEST(UnityLinearizationParametersTest, ResidualVector) {
    auto gen_coords = create_vector({1., 2., 3., 4., 5., 6., 7.});
    auto v = create_vector({1., 2., 3., 4., 5., 6.});
    auto velocity = v;
    auto acceleration = v;
    auto lagrange_mults = create_vector({1., 2., 3.});

    UnityLinearizationParameters unity_linearization_parameters;

    auto residual_vector = unity_linearization_parameters.ResidualVector(
        gen_coords, velocity, acceleration, lagrange_mults
    );

    expect_kokkos_view_1D_equal(
        residual_vector,
        {
            1., 1., 1., 1., 1., 1., 1., 1., 1.  // 9 elements
        }
    );
}

TEST(UnityLinearizationParametersTest, IterationMatrix) {
    auto gen_coords = create_vector({1., 2., 3., 4., 5., 6., 7.});
    auto v = create_vector({1., 2., 3., 4., 5., 6.});
    auto delta_gen_coords = v;
    auto velocity = v;
    auto acceleration = v;
    auto lagrange_mults = create_vector({1., 2., 3.});

    UnityLinearizationParameters unity_linearization_parameters;

    auto iteration_matrix = unity_linearization_parameters.IterationMatrix(
        1., 1., 1., gen_coords, delta_gen_coords, velocity, acceleration, lagrange_mults
    );

    expect_kokkos_view_2D_equal(
        iteration_matrix,
        {
            {1., 0., 0., 0., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0., 0., 0., 0.},  // row 3
            {0., 0., 0., 1., 0., 0., 0., 0., 0.},  // row 4
            {0., 0., 0., 0., 1., 0., 0., 0., 0.},  // row 5
            {0., 0., 0., 0., 0., 1., 0., 0., 0.},  // row 6
            {0., 0., 0., 0., 0., 0., 1., 0., 0.},  // row 7
            {0., 0., 0., 0., 0., 0., 0., 1., 0.},  // row 8
            {0., 0., 0., 0., 0., 0., 0., 0., 1.}   // row 9
        }
    );
}

}  // namespace openturbine::gen_alpha_solver::tests
