#include <gtest/gtest.h>

#include "src/gebt_poc/section.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(StiffnessMatrixTest, ConstructorWithProvidedZerosMatrix) {
    Kokkos::View<double**> matrix("stiffness_matrix", 6, 6);
    Kokkos::deep_copy(matrix, 0.);

    StiffnessMatrix stiffness(matrix);

    EXPECT_EQ(stiffness.GetStiffnessMatrix().extent(0), 6);
    EXPECT_EQ(stiffness.GetStiffnessMatrix().extent(1), 6);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        stiffness.GetStiffnessMatrix(),
        {
            {0., 0., 0., 0., 0., 0.},  // row 1
            {0., 0., 0., 0., 0., 0.},  // row 2
            {0., 0., 0., 0., 0., 0.},  // row 3
            {0., 0., 0., 0., 0., 0.},  // row 4
            {0., 0., 0., 0., 0., 0.},  // row 5
            {0., 0., 0., 0., 0., 0.}   // row 6
        }
    );
}

TEST(StiffnessMatrixTest, ConstructorWithProvidedRandomMatrix) {
    auto stiffness_matrix = StiffnessMatrix(gen_alpha_solver::create_matrix({
        {1., 0., 0., 0., 0., 0.},  // row 1
        {0., 2., 0., 0., 0., 0.},  // row 2
        {0., 0., 3., 0., 0., 0.},  // row 3
        {0., 0., 0., 4., 0., 0.},  // row 4
        {0., 0., 0., 0., 5., 0.},  // row 5
        {0., 0., 0., 0., 0., 6.}   // row 6
    }));

    StiffnessMatrix stiffness(stiffness_matrix);

    EXPECT_EQ(stiffness.GetStiffnessMatrix().extent(0), 6);
    EXPECT_EQ(stiffness.GetStiffnessMatrix().extent(1), 6);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        stiffness.GetStiffnessMatrix(),
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 2., 0., 0., 0., 0.},  // row 2
            {0., 0., 3., 0., 0., 0.},  // row 3
            {0., 0., 0., 4., 0., 0.},  // row 4
            {0., 0., 0., 0., 5., 0.},  // row 5
            {0., 0., 0., 0., 0., 6.}   // row 6
        }
    );
}

TEST(SectionTest, ConstructUnitSection) {
    auto mass_matrix = gen_alpha_solver::MassMatrix(1., 1.);
    auto stiffness_matrix = StiffnessMatrix(gen_alpha_solver::create_matrix({
        {1., 0., 0., 0., 0., 0.},  // row 1
        {0., 1., 0., 0., 0., 0.},  // row 2
        {0., 0., 1., 0., 0., 0.},  // row 3
        {0., 0., 0., 1., 0., 0.},  // row 4
        {0., 0., 0., 0., 1., 0.},  // row 5
        {0., 0., 0., 0., 0., 1.}   // row 6
    }));
    auto location = 0.;

    auto section = Section(location, mass_matrix, stiffness_matrix);

    EXPECT_EQ(section.GetNormalizedLocation(), location);
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        section.GetMass().GetMassMatrix(),
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0.},  // row 3
            {0., 0., 0., 1., 0., 0.},  // row 4
            {0., 0., 0., 0., 1., 0.},  // row 5
            {0., 0., 0., 0., 0., 1.}   // row 6
        }
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        section.GetStiffnessMatrix().GetStiffnessMatrix(),
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

TEST(SectionTest, ExpectThrowIfLocationIsOutOfBounds) {
    auto mass_matrix = gen_alpha_solver::MassMatrix(1., 1.);
    auto stiffness_matrix = StiffnessMatrix(gen_alpha_solver::create_matrix({
        {1., 0., 0., 0., 0., 0.},  // row 1
        {0., 1., 0., 0., 0., 0.},  // row 2
        {0., 0., 1., 0., 0., 0.},  // row 3
        {0., 0., 0., 1., 0., 0.},  // row 4
        {0., 0., 0., 0., 1., 0.},  // row 5
        {0., 0., 0., 0., 0., 1.}   // row 6
    }));
    auto location_less_than_zero = -0.1;
    auto location_greater_than_one = 1.1;

    EXPECT_THROW(
        Section(location_less_than_zero, mass_matrix, stiffness_matrix), std::invalid_argument
    );
    EXPECT_THROW(
        Section(location_greater_than_one, mass_matrix, stiffness_matrix), std::invalid_argument
    );
}

}  // namespace openturbine::gebt_poc::tests
