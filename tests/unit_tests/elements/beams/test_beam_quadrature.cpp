#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include "elements/beams/beam_quadrature.hpp"

namespace kynema::beams::tests {

TEST(BeamQuadratureTest, CheckCreateTrapezoidalQuadrature_1) {
    const auto q_act = CreateTrapezoidalQuadrature(std::array{0., 0.2, 0.4, 0.6, 0.8, 1.0});
    constexpr auto q_exp =
        std::array<std::array<double, 2>, 6>{std::array{-1., 0.2}, {-0.6, 0.4}, {-0.2, 0.4},
                                             {0.2, 0.4},           {0.6, 0.4},  {1., 0.2}};
    for (auto i : std::views::iota(0U, 6U)) {
        for (auto j : std::views::iota(0U, 2U)) {
            EXPECT_NEAR(q_act[i][j], q_exp[i][j], 1e-14);
        }
    }
}

TEST(BeamQuadratureTest, CheckCreateTrapezoidalQuadrature_2) {
    const auto q_act = CreateTrapezoidalQuadrature(std::array{-5., -3., -1., 0., 3., 4., 5.});
    constexpr auto q_exp = std::array<std::array<double, 2>, 7>{
        std::array{-1., 0.2}, {-0.6, 0.4}, {-0.2, 0.3}, {0., 0.4},
        {0.6, 0.4},           {0.8, 0.2},  {1., 0.1}
    };
    for (auto i : std::views::iota(0U, 7U)) {
        for (auto j : std::views::iota(0U, 2U)) {
            EXPECT_NEAR(q_act[i][j], q_exp[i][j], 1e-14);
        }
    }
}

}  // namespace kynema::beams::tests
