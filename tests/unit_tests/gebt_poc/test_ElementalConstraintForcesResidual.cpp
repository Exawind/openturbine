#include <gtest/gtest.h>

#include "src/gebt_poc/ElementalConstraintForcesResidual.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

TEST(SolverTest, ElementalConstraintForcesResidual) {
    auto generalized_coords = gen_alpha_solver::create_matrix(
        {{0.1, 0., 0.12, 0.9987502603949662, 0.049979169270678324, 0., 0.}}
    );

    auto constraints_residual = Kokkos::View<double[6]>("constraints_residual");

    ElementalConstraintForcesResidual(generalized_coords, constraints_residual);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        // constraints_residual should be same as the generalized_coords where
        // q{0.9987502603949662, 0.049979169270678324, 0., 0.} -> v{0.1, 0., 0.}
        constraints_residual, {0.1, 0., 0.12, 0.1, 0., 0.0}
    );
}

TEST(SolverTest, TestAxialVectorOfARotationMatrix) {
    auto matrix = gen_alpha_solver::create_matrix({{1., 0., 0.}, {0., 0., -1.}, {0., 1., 0.}});
    auto axial_vector = Kokkos::View<double[3]>("axial_vector");
    AxialVector(matrix, axial_vector);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(axial_vector, {1., 0., 0.});
}

TEST(SolverTest, TestConstraintResidualForRotatingBeam) {
    auto initial_position = gen_alpha_solver::create_matrix({
        {2., 0., 0., 1., 0., 0., 0.},                  // node 1
        {3.7267316464601146, 0., 0., 1., 0., 0., 0.},  // node 2
        {7., 0., 0., 1., 0., 0., 0.},                  // node 3
        {10.273268353539885, 0., 0., 1., 0., 0., 0.},  // node 4
        {12., 0., 0., 1., 0., 0., 0.}                  // node 5
    });
    auto gen_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},  // node 1
        {0., 0., 0., 1., 0., 0., 0.},  // node 2
        {0., 0., 0., 1., 0., 0., 0.},  // node 3
        {0., 0., 0., 1., 0., 0., 0.},  // node 4
        {0., 0., 0., 1., 0., 0., 0.}   // node 5
    });
    auto applied_motion = gen_alpha_solver::create_vector({0., 0., 0., 0.999997, 0., 0., 0.0025});
    auto constraints_residual = Kokkos::View<double[6]>("constraints_residual");

    ConstraintResidualForRotatingBeam(
        initial_position, gen_coords, applied_motion, constraints_residual
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        constraints_residual,
        {2.4999947916715115e-5, -0.009999958333385416, 0.0, 0.0, 0.0, -0.004999979166692708}
    );
}

}  // namespace openturbine::gebt_poc
