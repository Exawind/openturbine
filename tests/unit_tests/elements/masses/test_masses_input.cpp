#include <gtest/gtest.h>

#include "src/elements/masses/masses_input.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

class MassesInputTest : public ::testing::Test {
protected:
    static auto CreateTestElements() {
        // Create a mock Model and add 2 nodes
        auto model = Model();
        std::vector<size_t> node_ids{
            model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build(),
            model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build(),
        };

        // Create an identity mass matrix
        constexpr auto mass_matrix = std::array{
            std::array{1., 0., 0., 0., 0., 0.},  //
            std::array{0., 1., 0., 0., 0., 0.},  //
            std::array{0., 0., 1., 0., 0., 0.},  //
            std::array{0., 0., 0., 1., 0., 0.},  //
            std::array{0., 0., 0., 0., 1., 0.},  //
            std::array{0., 0., 0., 0., 0., 1.}   //
        };

        // Create 2 mass elements
        return std::vector<MassElement>{
            MassElement(0U, 0U, mass_matrix),  // element 1
            MassElement(1U, 1U, mass_matrix)   // element 2
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
