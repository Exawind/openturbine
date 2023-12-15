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

TEST(MassMatrixTest, CreateMassMatrixWithGivenProperties) {
    auto mass = 2.;
    auto center_of_mass = gen_alpha_solver::create_vector({0.1, 0.2, 0.3});
    auto moment_of_inertia =
        gen_alpha_solver::create_matrix({{1., 2., 3.}, {2., 4., 6.}, {3., 6., 9.}});

    auto mass_matrix = MassMatrix(mass, center_of_mass, moment_of_inertia);

    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {2., 0., 0., 0., 0.6, -0.4},  // row 1
            {0., 2., 0., -0.6, 0., 0.2},  // row 2
            {0., 0., 2., 0.4, -0.2, 0.},  // row 3
            {0., -0.6, 0.4, 1., 2., 3.},  // row 4
            {0.6, 0., -0.2, 2., 4., 6.},  // row 5
            {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
        }
    );
}

TEST(MassMatrixTest, CreateMassMatrixWithGivenMatrix) {
    auto mm = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    });

    auto mass_matrix = MassMatrix(mm);

    EXPECT_EQ(mass_matrix.GetMass(), 2.);
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        mass_matrix.GetCenterOfMass(),
        {
            0.1, 0.2, 0.3  // 3 elements
        }
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        mass_matrix.GetMomentOfInertia(),
        {
            {1., 2., 3.},  // row 1
            {2., 4., 6.},  // row 2
            {3., 6., 9.}   // row 3
        }
    );
}

}  // namespace openturbine::gebt_poc::tests
