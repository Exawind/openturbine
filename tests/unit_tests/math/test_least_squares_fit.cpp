#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/math/least_squares_fit.hpp"

namespace openturbine::tests {

TEST(LeastSquaresFitTest, MapGeometricLocations_PositiveRange) {
    std::vector<double> input = {0., 2.5, 5.};
    std::vector<double> expected = {-1., 0., 1.};

    auto result = openturbine::MapGeometricLocations(input);

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1.e-15);
    }
}

TEST(LeastSquaresFitTest, MapGeometricLocations_NegativeRange) {
    std::vector<double> input = {-10., -5., 0.};
    std::vector<double> expected = {-1., 0., 1.};

    auto result = openturbine::MapGeometricLocations(input);

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1.e-15);
    }
}

TEST(LeastSquaresFitTest, MapGeometricLocations_UnitRange) {
    std::vector<double> input = {0., 0.5, 1.};
    std::vector<double> expected = {-1., 0., 1.};

    auto result = openturbine::MapGeometricLocations(input);

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1.e-15);
    }
}

TEST(LeastSquaresFitTest, MapGeometricLocations_InvalidInput) {
    std::vector<double> input = {1., 1.};

    EXPECT_THROW(openturbine::MapGeometricLocations(input), std::invalid_argument);
}

}  // namespace openturbine::tests
