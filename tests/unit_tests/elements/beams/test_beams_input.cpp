#include <gtest/gtest.h>

#include "src/elements/beams/beams_input.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

class BeamsInputTest : public ::testing::Test {
protected:
    static auto CreateTestElements() {
        // Create mass and stiffness matrices (identity matrices)
        constexpr auto mass_matrix = std::array{
            std::array{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  //
            std::array{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  //
            std::array{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},  //
            std::array{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},  //
            std::array{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  //
            std::array{0.0, 0.0, 0.0, 0.0, 0.0, 1.0},  //
        };

        constexpr auto stiffness_matrix = std::array{
            std::array{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  //
            std::array{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  //
            std::array{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},  //
            std::array{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},  //
            std::array{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  //
            std::array{0.0, 0.0, 0.0, 0.0, 0.0, 1.0},  //
        };

        // Create a mock Model for creating nodes
        auto model = Model();
        std::vector<double> s{0., 0.5, 1.0, 0., 1., 0., 0.33, 0.67, 1.0};
        for (size_t i = 0; i < 9; ++i) {
            model.AddNode()
                .SetElemLocation(s[i])
                .SetPosition(static_cast<double>(i), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                .Build();
        }

        // Create 3 elements with different numbers of nodes and quadrature points
        // Element 1 - 3 nodes, 2 quadrature points
        model.AddBeamElement(
            {0U, 1U, 2U},
            {
                BeamSection(0.0, mass_matrix, stiffness_matrix),
                BeamSection(1.0, mass_matrix, stiffness_matrix),
            },
            BeamQuadrature{{-0.5, 0.5}, {0.5, 0.5}}
        );
        // Element 2 - 2 nodes, 2 quadrature points
        model.AddBeamElement(
            {3, 4},
            {
                BeamSection(0.0, mass_matrix, stiffness_matrix),
                BeamSection(1.0, mass_matrix, stiffness_matrix),
            },
            BeamQuadrature{{-0.5773502691896257, 1.0}, {0.5773502691896257, 1.0}}
        );
        // Element 3 - 4 nodes, 4 quadrature points
        model.AddBeamElement(
            {5, 6, 7, 8},
            {
                BeamSection(0.0, mass_matrix, stiffness_matrix),
                BeamSection(1.0, mass_matrix, stiffness_matrix),
            },
            BeamQuadrature{
                {-0.861136311594053, 0.347854845137454},
                {-0.339981043584856, 0.652145154862546},
                {0.339981043584856, 0.652145154862546},
                {0.861136311594053, 0.347854845137454}
            }
        );

        return model.GetBeamElements();
    }

    static BeamsInput CreateTestBeamsInput() {
        return BeamsInput(CreateTestElements(), {0.0, 0.0, -9.81});
    }
};

TEST_F(BeamsInputTest, Constructor) {
    const auto elements = CreateTestElements();
    const std::array<double, 3> gravity = {0.0, 0.0, -9.81};
    const BeamsInput input(elements, gravity);

    EXPECT_EQ(input.elements.size(), 3);
    EXPECT_EQ(input.gravity, gravity);
}

TEST_F(BeamsInputTest, NumElements) {
    const auto input = CreateTestBeamsInput();
    // 3 elements
    EXPECT_EQ(input.NumElements(), 3);
}

TEST_F(BeamsInputTest, NumNodes) {
    const auto input = CreateTestBeamsInput();
    // 9 nodes across all elements
    EXPECT_EQ(input.NumNodes(), 9);
}

TEST_F(BeamsInputTest, NumQuadraturePoints) {
    const auto input = CreateTestBeamsInput();
    // 8 quadrature points across all elements
    EXPECT_EQ(input.NumQuadraturePoints(), 8);
}

TEST_F(BeamsInputTest, MaxElemNodes) {
    const auto input = CreateTestBeamsInput();
    // Element 3 has the most nodes (4)
    EXPECT_EQ(input.MaxElemNodes(), 4);
}

TEST_F(BeamsInputTest, MaxElemQuadraturePoints) {
    const auto input = CreateTestBeamsInput();
    // Element 3 has the most quadrature points (4)
    EXPECT_EQ(input.MaxElemQuadraturePoints(), 4);
}

}  // namespace openturbine::tests
