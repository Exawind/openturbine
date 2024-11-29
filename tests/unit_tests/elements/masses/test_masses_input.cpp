#include <gtest/gtest.h>

#include "src/elements/masses/masses_input.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

class MassesInputTest : public ::testing::Test {
protected:
    static auto CreateTestElements() {
        // Create a mock Model for creating nodes
        auto model = Model();
        for (int i = 0; i < 2; ++i) {
            model.AddNode(
                {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
            );
        }

        // Create mass matrix (identity matrix)
        constexpr auto mass_matrix = std::array{
            std::array{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, std::array{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            std::array{0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            std::array{0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, std::array{0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
        };

        return std::vector<MassElement>{
            MassElement(model.GetNode(0), mass_matrix), MassElement(model.GetNode(1), mass_matrix)
        };
    }

    static MassesInput CreateTestMassesInput() {
        return MassesInput(CreateTestElements(), {0.0, 0.0, -9.81});
    }
};

TEST_F(MassesInputTest, Constructor) {
    const auto elements = CreateTestElements();
    const std::array<double, 3> gravity = {0.0, 0.0, -9.81};
    const MassesInput input(elements, gravity);

    EXPECT_EQ(input.elements.size(), 2);
    EXPECT_EQ(input.gravity, gravity);
}

TEST_F(MassesInputTest, NumElements) {
    const auto input = CreateTestMassesInput();
    EXPECT_EQ(input.NumElements(), 2);
}

}  // namespace openturbine::tests
