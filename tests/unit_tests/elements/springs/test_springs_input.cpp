#include <gtest/gtest.h>

#include "src/elements/springs/springs_input.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

class SpringsInputTest : public ::testing::Test {
protected:
    static auto CreateTestElements() {
        auto model = Model();
        // Add 4 nodes to the model
        for (int i = 0; i < 4; ++i) {
            model.AddNode().SetPosition(static_cast<double>(i), 0., 0., 1., 0., 0., 0.).Build();
        }
        // Create 2 spring elements based on the nodes
        return std::vector<SpringElement>{
            SpringElement(0U, {0U, 1U}, 1000., 0.),  // element 1
            SpringElement(1U, {2U, 3U}, 2000., 0.)   // element 2
        };
    }

    static SpringsInput CreateTestSpringsInput() { return SpringsInput{CreateTestElements()}; }
};

TEST_F(SpringsInputTest, Constructor) {
    const auto elements = CreateTestElements();
    const SpringsInput input(elements);
    EXPECT_EQ(input.elements.size(), 2);
}

TEST_F(SpringsInputTest, NumElements) {
    const auto input = CreateTestSpringsInput();
    EXPECT_EQ(input.NumElements(), 2);
}

}  // namespace openturbine::tests
