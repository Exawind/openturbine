#include <gtest/gtest.h>

#include "src/gebt_poc/model.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(ModelTest, ConstructDefaultModel) {
    Model model("model");

    EXPECT_EQ(model.GetName(), "model");
    EXPECT_EQ(model.GetSections().size(), 0);
}

TEST(ModelTest, AddASectionToModel) {
    Model model("model");
    auto section = Section(
        0., gen_alpha_solver::MassMatrix(1., 1.), gen_alpha_solver::create_identity_matrix(6)
    );

    model.AddSection(section);

    EXPECT_EQ(model.GetSections().size(), 1);
}

}  // namespace openturbine::gebt_poc::tests
