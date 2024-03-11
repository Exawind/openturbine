#include <gtest/gtest.h>

#include "src/gebt_poc/MassMatrix.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

TEST(MassMatrixTest, GetMass) {
    auto mass_matrix = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    });

    EXPECT_EQ(GetMass(mass_matrix), 2.);
}

TEST(MassMatrixTest, GetCenterOfMass) {
    auto mass_matrix = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    });

    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        GetCenterOfMass(mass_matrix), {0.1, 0.2, 0.3}
    );
}

TEST(MassMatrixTest, GetMomentOfInertia) {
    auto mass_matrix = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    });

    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        GetMomentOfInertia(mass_matrix),
        {
            {1., 2., 3.},  // row 1
            {2., 4., 6.},  // row 2
            {3., 6., 9.}   // row 3
        }
    );
}

}  // namespace openturbine::gebt_poc