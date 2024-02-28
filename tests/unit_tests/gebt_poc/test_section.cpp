#include <gtest/gtest.h>

#include "src/gebt_poc/section.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(StiffnessMatrixTest, DefaultConstructor) {
    StiffnessMatrix stiffness;

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        stiffness.GetStiffnessMatrix(),
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

TEST(StiffnessMatrixTest, ConstructorWithProvidedZerosMatrix) {
    Kokkos::View<double**> matrix("stiffness_matrix", 6, 6);
    Kokkos::deep_copy(matrix, 0.);

    StiffnessMatrix stiffness(matrix);

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

TEST(SectionTest, DefaultConstructor) {
    Section section;

    EXPECT_EQ(section.GetName(), "");
    EXPECT_EQ(section.GetNormalizedLocation(), 0.);
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        section.GetMassMatrix().GetMassMatrix(),
        {
            {0., 0., 0., 0., 0., 0.},  // row 1
            {0., 0., 0., 0., 0., 0.},  // row 2
            {0., 0., 0., 0., 0., 0.},  // row 3
            {0., 0., 0., 0., 0., 0.},  // row 4
            {0., 0., 0., 0., 0., 0.},  // row 5
            {0., 0., 0., 0., 0., 0.}   // row 6
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

TEST(SectionTest, ConstructUnitSection) {
    auto mass = MassMatrix(gen_alpha_solver::create_identity_matrix(6));
    auto stiffness = StiffnessMatrix(gen_alpha_solver::create_identity_matrix(6));
    auto location = 0.;

    auto section = Section("section_1", location, mass, stiffness);

    EXPECT_EQ(section.GetNormalizedLocation(), location);
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        section.GetMassMatrix().GetMassMatrix(),
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
    EXPECT_EQ(section.GetName(), "section_1");
}

TEST(SectionTest, ExpectThrowIfLocationIsOutOfBounds) {
    auto name = "section_1";
    auto mass = MassMatrix(gen_alpha_solver::create_identity_matrix(6));
    auto stiffness = StiffnessMatrix(gen_alpha_solver::create_identity_matrix(6));
    auto location_less_than_zero = -0.1;
    auto location_greater_than_one = 1.1;

    EXPECT_THROW(Section(name, location_less_than_zero, mass, stiffness), std::invalid_argument);
    EXPECT_THROW(Section(name, location_greater_than_one, mass, stiffness), std::invalid_argument);
}

}  // namespace openturbine::gebt_poc::tests
