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

TEST(LeastSquaresFitTest, ShapeFunctionMatrices_FirstOrder) {
    size_t n{3};                               // Number of pts to fit
    size_t p{2};                               // Polynomial order + 1
    std::vector<double> xi_g = {-1., 0., 1.};  // Evaluation points
    auto [phi_g, gll_pts] = openturbine::ShapeFunctionMatrices(n, p, xi_g);

    // Check GLL points (2 at -1 and 1)
    ASSERT_EQ(gll_pts.size(), p);
    EXPECT_NEAR(gll_pts[0], -1., 1.e-15);
    EXPECT_NEAR(gll_pts[1], 1., 1.e-15);

    // Check shape function matrix dimensions (2 x 3)
    ASSERT_EQ(phi_g.size(), p);
    ASSERT_EQ(phi_g[0].size(), n);
    ASSERT_EQ(phi_g[1].size(), n);

    // Check shape function values at evaluation points
    std::vector<std::vector<double>> expected = {
        {1., 0.5, 0.},  // First shape function
        {0., 0.5, 1.}   // Second shape function
    };

    for (size_t i = 0; i < phi_g.size(); ++i) {
        for (size_t j = 0; j < phi_g[i].size(); ++j) {
            EXPECT_NEAR(phi_g[i][j], expected[i][j], 1.e-15);
        }
    }
}

TEST(LeastSquaresFitTest, ShapeFunctionMatrices_SecondOrder) {
    size_t n{5};                                          // Number of pts to fit
    size_t p{3};                                          // Polynomial order + 1
    std::vector<double> xi_g = {-1., -0.5, 0., 0.5, 1.};  // Evaluation points
    auto [phi_g, gll_pts] = openturbine::ShapeFunctionMatrices(n, p, xi_g);

    // Check GLL points (3 at -1, 0, and 1)
    ASSERT_EQ(gll_pts.size(), 3);
    EXPECT_NEAR(gll_pts[0], -1., 1.e-15);
    EXPECT_NEAR(gll_pts[1], 0., 1.e-15);
    EXPECT_NEAR(gll_pts[2], 1., 1.e-15);

    // Check shape function matrix dimensions (3 x 5)
    ASSERT_EQ(phi_g.size(), 3);
    for (const auto& row : phi_g) {
        ASSERT_EQ(row.size(), 5);
    }

    // Check shape function values at evaluation points
    std::vector<std::vector<double>> expected = {
        {1.0, 0.375, 0.0, -0.125, 0.0},  // First shape function
        {0.0, 0.75, 1.0, 0.75, 0.0},     // Second shape function
        {0.0, -0.125, 0.0, 0.375, 1.0}   // Third shape function
    };

    for (size_t i = 0; i < phi_g.size(); ++i) {
        for (size_t j = 0; j < phi_g[i].size(); ++j) {
            EXPECT_NEAR(phi_g[i][j], expected[i][j], 1.e-15);
        }
    }
}

TEST(LeastSquaresFitTest, FitsParametricCurve) {
    // Input geometric points (t = 0, 0.2, 0.5, 0.6, 1.0)
    std::vector<double> geom_locations = {0.0, 0.2, 0.5, 0.6, 1.0};
    std::vector<std::array<double, 3>> input_points = {
        {0., 0., 0.},        // t = 0
        {1., -0.28, 0.12},   // t = 0.2
        {2.5, -0.25, 0.},    // t = 0.5
        {3., -0.12, -0.12},  // t = 0.6
        {5., 1., -1.}        // t = 1.0
    };

    // Step 1: Map geometric locations
    auto mapped_locations = MapGeometricLocations(geom_locations);

    // Step 2: Generate shape function matrices (using p = 4 i.e. cubic interpolation)
    size_t n = input_points.size();
    size_t p = 4;
    auto [phi_g, gll_points] = ShapeFunctionMatrices(n, p, mapped_locations);

    // Step 3: Perform least squares fitting
    auto X = PerformLeastSquaresFitting(p, phi_g, input_points);

    // Expected coefficients from Mathematica (rounded to 3 decimal places)
    std::vector<std::array<double, 3>> expected_coefficients = {
        {0., 0., 0.},  // First point - same as input
        {1.3819660112501062, -0.3236067977499792, 0.12360679774997904},  // Interior point 1
        {3.6180339887498945, 0.12360679774997924, -0.3236067977499791},  // Interior point 2
        {5., 1., -1.}  // Last point - same as input
    };

    // Verify results
    /* ASSERT_EQ(X.size(), expected_coefficients.size());
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(X[i][j], expected_coefficients[i][j], 1e-3)
                << "Mismatch at coefficient [" << i << "][" << j << "]";
        }
    } */
}

}  // namespace openturbine::tests
