#include <gtest/gtest.h>

#include "src/gebt_poc/model.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(ModelTest, ConstructModelWithNoSections) {
    Model model("model", Kokkos::View<Section*>("sections", 0));

    EXPECT_EQ(model.GetName(), "model");
    EXPECT_EQ(model.GetSections().extent(0), 0);
}

TEST(ModelTest, ConstructModelWithOneSection) {
    auto section = Section(0.5, gen_alpha_solver::MassMatrix(), StiffnessMatrix(), "section_1");
    Kokkos::View<Section*> sections("sections", 1);
    Kokkos::deep_copy(sections, section);

    Model model("model", sections);

    EXPECT_EQ(model.GetName(), "model");
    EXPECT_EQ(model.GetSections().extent(0), 1);
    EXPECT_EQ(model.GetSections()(0).GetName(), "section_1");
}

}  // namespace openturbine::gebt_poc::tests
