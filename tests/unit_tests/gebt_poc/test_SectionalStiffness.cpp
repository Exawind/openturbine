#include <gtest/gtest.h>

#include "src/gebt_poc/SectionalStiffness.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

TEST(SolverTest, SectionalStiffness) {
    auto rotation_0 = gen_alpha_solver::create_matrix({
        {1., 2., 3.},  // row 1
        {4., 5., 6.},  // row 2
        {7., 8., 9.}   // row 3
    });
    auto rotation = gen_alpha_solver::create_matrix({
        {1., 0., 0.},  // row 1
        {0., 1., 0.},  // row 2
        {0., 0., 1.}   // row 3
    });

    auto stiffness = gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    });

    auto sectional_stiffness = Kokkos::View<double[6][6]>("sectional_stiffness");
    SectionalStiffness(stiffness, rotation_0, rotation, sectional_stiffness);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        sectional_stiffness,
        {
            {196., 448., 700., 448., 1078., 1708.},      // row 1
            {448., 1024., 1600., 1024., 2464., 3904.},   // row 2
            {700., 1600., 2500., 1600., 3850., 6100.},   // row 3
            {448., 1024., 1600., 1024., 2464., 3904.},   // row 4
            {1078., 2464., 3850., 2464., 5929., 9394.},  // row 5
            {1708., 3904., 6100., 3904., 9394., 14884.}  // row 6
        }
    );
}

}  // namespace openturbine::gebt_poc