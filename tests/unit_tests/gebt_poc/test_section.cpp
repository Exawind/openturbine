#include <gtest/gtest.h>

#include "src/gebt_poc/section.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(SectionTest, DefaultConstructor) {
    Section section;

    EXPECT_EQ(section.GetName(), "");
    EXPECT_EQ(section.GetNormalizedLocation(), 0.);
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        section.GetMassMatrix(),
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
        section.GetStiffnessMatrix(),
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

TEST(SectionTest, ConstructUnitSection) {
    auto mass = gen_alpha_solver::create_identity_matrix(6);
    auto stiffness = gen_alpha_solver::create_identity_matrix(6);
    auto location = 0.;

    auto section = Section("section_1", location, mass, stiffness);

    EXPECT_EQ(section.GetNormalizedLocation(), location);
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        section.GetMassMatrix(),
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
        section.GetStiffnessMatrix(),
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
    auto mass = gen_alpha_solver::create_identity_matrix(6);
    auto stiffness = gen_alpha_solver::create_identity_matrix(6);
    auto location_less_than_zero = -0.1;
    auto location_greater_than_one = 1.1;

    EXPECT_THROW(Section(name, location_less_than_zero, mass, stiffness), std::invalid_argument);
    EXPECT_THROW(Section(name, location_greater_than_one, mass, stiffness), std::invalid_argument);
}

}  // namespace openturbine::gebt_poc::tests
