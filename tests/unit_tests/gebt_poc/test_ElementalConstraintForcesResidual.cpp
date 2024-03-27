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
    auto matrix = gen_alpha_solver::create_matrix(
        {{1., 0., 0.}, {0., 0., -1.}, {0., 1., 0.}}
    );
    auto axial_vector = Kokkos::View<double[3]>("axial_vector");
    AxialVector(matrix, axial_vector);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        axial_vector, {1., 0., 0.}
    );
}

}  // namespace openturbine::gebt_poc
