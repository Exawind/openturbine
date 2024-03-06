#include <gtest/gtest.h>

#include "src/gebt_poc/model.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(ModelTest, ConstructDefaultModel) {
    Model model("model");

    EXPECT_EQ(model.GetName(), "model");
    EXPECT_EQ(model.GetSections().size(), 0);
}

TEST(ModelTest, ConstructModelWithSections) {
    auto section_1 = Section();
    auto section_2 = Section();

    Model model("model", {section_1, section_2});

    EXPECT_EQ(model.GetName(), "model");
    EXPECT_EQ(model.GetSections().size(), 2);
}

TEST(ModelTest, AddASectionToModel) {
    Model model("model");
    auto mass = MassMatrix(gen_alpha_solver::create_identity_matrix(6));
    auto stiffness = gen_alpha_solver::create_identity_matrix(6);
    auto section = Section("s_1", 0., mass, stiffness);

    model.AddSection(section);

    EXPECT_EQ(model.GetSections().size(), 1);
}

}  // namespace openturbine::gebt_poc::tests
