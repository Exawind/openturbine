#include <gtest/gtest.h>

#include "src/gebt_poc/quadrature.h"

namespace openturbine::gebt_poc {
TEST(SolverTest, UserDefinedQuadrature) {
    auto quadrature_points = std::vector<double>{
        -0.9491079123427585,  // point 1
        -0.7415311855993945,  // point 2
        -0.4058451513773972,  // point 3
        0.,                   // point 4
        0.4058451513773972,   // point 5
        0.7415311855993945,   // point 6
        0.9491079123427585    // point 7
    };
    auto quadrature_weights = std::vector<double>{
        0.1294849661688697,  // weight 1
        0.2797053914892766,  // weight 2
        0.3818300505051189,  // weight 3
        0.4179591836734694,  // weight 4
        0.3818300505051189,  // weight 5
        0.2797053914892766,  // weight 6
        0.1294849661688697   // weight 7
    };
    auto quadrature = UserDefinedQuadrature(quadrature_points, quadrature_weights);

    EXPECT_EQ(quadrature.GetNumberOfQuadraturePoints(), 7);
    EXPECT_EQ(quadrature.GetQuadraturePoints(), quadrature_points);
    EXPECT_EQ(quadrature.GetQuadratureWeights(), quadrature_weights);
}
}  // namespace openturbine::gebt_poc