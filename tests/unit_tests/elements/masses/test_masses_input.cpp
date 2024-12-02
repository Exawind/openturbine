#include <gtest/gtest.h>

#include "src/elements/masses/masses_input.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

class MassesInputTest : public ::testing::Test {
protected:
    static auto CreateTestElements() {
        // Create a mock Model and add 2 nodes
        auto model = Model();
        for (int i = 0; i < 2; ++i) {
            model.AddNode(
                {0., 0., 0., 1., 0., 0., 0.}, {0., 0., 0., 1., 0., 0., 0.}, {0., 0., 0., 0., 0., 0.},
                {0., 0., 0., 0., 0., 0.}
            );
        }

        // Create an identity mass matrix
        constexpr auto mass_matrix =
            std::array{std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
                       std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
                       std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.}};

        // Create 2 mass elements
        return std::vector<MassElement>{
            MassElement(model.GetNode(0), mass_matrix),  // element 1
            MassElement(model.GetNode(1), mass_matrix)   // element 2
        };
    }

    static MassesInput CreateTestMassesInput() {
        return MassesInput(CreateTestElements(), {0., 0., -9.81});
    }
};

TEST_F(MassesInputTest, Constructor) {
    const auto elements = CreateTestElements();
    const std::array<double, 3> gravity = {0., 0., -9.81};
    const MassesInput input(elements, gravity);

    EXPECT_EQ(input.elements.size(), 2);
    EXPECT_EQ(input.gravity, gravity);
}

TEST_F(MassesInputTest, NumElements) {
    const auto input = CreateTestMassesInput();
    EXPECT_EQ(input.NumElements(), 2);
}

}  // namespace openturbine::tests
