#include <gtest/gtest.h>

#include "src/gebt_poc/model.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(ModelTest, ConstructModelWithZeroSections) {
    Model model("model", CreateSections({}));

    EXPECT_EQ(model.GetName(), "model");
    EXPECT_EQ(model.GetSections().extent(0), 0);
}

TEST(ModelTest, ConstructModelWithOneSection) {
    Model model("model", CreateSections({Section()}));

    EXPECT_EQ(model.GetName(), "model");
    EXPECT_EQ(model.GetSections().extent(0), 1);
}

}  // namespace openturbine::gebt_poc::tests
