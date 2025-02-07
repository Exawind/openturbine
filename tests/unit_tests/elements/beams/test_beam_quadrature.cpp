#include <gtest/gtest.h>

#include "elements/beams/beam_quadrature.hpp"

namespace openturbine::tests {

TEST(BeamQuadratureTest, CheckCreateTrapezoidalQuadrature) {
    struct TestData {
        std::vector<double> grid;
        BeamQuadrature q_exp;
    };

    std::vector<TestData> test_data{
        {
            {0., 0.2, 0.4, 0.6, 0.8, 1.0},  // Grid
            {
                {-1., 0.2},
                {-0.6, 0.4},
                {-0.2, 0.4},
                {0.2, 0.4},
                {0.6, 0.4},
                {1., 0.2},
            },  // Quadrature
        },
        {
            {-5., -3., -1., 0., 3., 4., 5.},  // Grid
            {
                {-1., 0.2},
                {-0.6, 0.4},
                {-0.2, 0.3},
                {0., 0.4},
                {0.6, 0.4},
                {0.8, 0.2},
                {1., 0.1},
            },  // Quadrature
        },
    };

    for (const TestData& td : test_data) {
        const auto q_act = CreateTrapezoidalQuadrature(td.grid);
        for (auto i = 0U; i < td.q_exp.size(); ++i) {
            for (auto j = 0U; j < 2U; ++j) {
                EXPECT_NEAR(q_act[i][j], td.q_exp[i][j], 1e-14);
            }
        }
    }
}

}  // namespace openturbine::tests
