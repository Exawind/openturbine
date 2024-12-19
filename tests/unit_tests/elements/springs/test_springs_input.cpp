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
            model.AddNode(
                {static_cast<double>(i), 0., 0., 1., 0., 0., 0.}, {0., 0., 0., 1., 0., 0., 0.},
                {0., 0., 0., 0., 0., 0.}, {0., 0., 0., 0., 0., 0.}
            );
        }
        // Create 2 spring elements based on the nodes
        return std::vector<SpringElement>{
            SpringElement(std::array{model.GetNode(0), model.GetNode(1)}, 1000.),
            SpringElement(std::array{model.GetNode(2), model.GetNode(3)}, 2000.)
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
