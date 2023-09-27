#include <gtest/gtest.h>

#include "src/gebt_poc/interpolation.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(MatrixInterpolationTest, FindNearestNeighborOnALine) {
    std::vector<Point> points = {
        Point(0., 0., 0.),  // point 1
        Point(1., 0., 0.),  // point 2
        Point(2., 0., 0.),  // point 3
        Point(3., 0., 0.),  // point 4
        Point(4., 0., 0.),  // point 5
        Point(5., 0., 0.),  // point 6
        Point(6., 0., 0.),  // point 7
        Point(7., 0., 0.),  // point 8
        Point(8., 0., 0.),  // point 9
        Point(9., 0., 0.)   // point 10
    };

    Point pt_1(1.5, 0., 0.);
    auto nearest_neighbor_1 = FindNearestNeighbor(points, pt_1);
    EXPECT_EQ(nearest_neighbor_1, Point(1., 0., 0.));

    Point pt_2(5.51, 0., 0.);
    auto nearest_neighbor_2 = FindNearestNeighbor(points, pt_2);
    EXPECT_EQ(nearest_neighbor_2, Point(6., 0., 0.));

    Point pt_3(8.49, 0., 0.);
    auto nearest_neighbor_3 = FindNearestNeighbor(points, pt_3);
    EXPECT_EQ(nearest_neighbor_3, Point(8., 0., 0.));

    Point pt_4(19.5, 0., 0.);
    auto nearest_neighbor_4 = FindNearestNeighbor(points, pt_4);
    EXPECT_EQ(nearest_neighbor_4, Point(9., 0., 0.));
}

TEST(MatrixInterpolationTest, FindNearestNeighborOnAPlane) {
    std::vector<Point> points = {
        Point(0., 0., 0.),  // point 1
        Point(1., 0., 0.),  // point 2
        Point(1., 1., 0.),  // point 3
        Point(0., 1., 0.),  // point 4
    };

    Point pt_1(0.5, 0.5, 0.);
    auto nearest_neighbor_1 = FindNearestNeighbor(points, pt_1);
    EXPECT_EQ(nearest_neighbor_1, Point(0., 0., 0.));

    Point pt_2(0.5, 0.51, 0.);
    auto nearest_neighbor_2 = FindNearestNeighbor(points, pt_2);
    EXPECT_EQ(nearest_neighbor_2, Point(1., 1., 0.));

    Point pt_3(0., 0.51, 0.);
    auto nearest_neighbor_3 = FindNearestNeighbor(points, pt_3);
    EXPECT_EQ(nearest_neighbor_3, Point(0., 1., 0.));

    Point pt_4(5., 5., 0.);
    auto nearest_neighbor_4 = FindNearestNeighbor(points, pt_4);
    EXPECT_EQ(nearest_neighbor_4, Point(1., 1., 0.));
}

TEST(MatrixInterpolationTest, Find2NearestNeighborOnALine) {
    std::vector<Point> points = {
        Point(0., 0., 0.),  // point 1
        Point(1., 0., 0.),  // point 2
        Point(2., 0., 0.),  // point 3
        Point(3., 0., 0.),  // point 4
        Point(4., 0., 0.),  // point 5
        Point(5., 0., 0.),  // point 6
        Point(6., 0., 0.),  // point 7
        Point(7., 0., 0.),  // point 8
        Point(8., 0., 0.),  // point 9
        Point(9., 0., 0.)   // point 10
    };

    Point pt_1(1.5, 0., 0.);
    auto nearest_neighbors_1 = FindkNearestNeighbors(points, pt_1, 2);
    std::vector<Point> expected_1 = {Point(1., 0., 0.), Point(2., 0., 0.)};
    EXPECT_EQ(nearest_neighbors_1, expected_1);

    Point pt_2(5.51, 0., 0.);
    auto nearest_neighbors_2 = FindkNearestNeighbors(points, pt_2, 2);
    std::vector<Point> expected_2 = {Point(6., 0., 0.), Point(5., 0., 0.)};
    EXPECT_EQ(nearest_neighbors_2, expected_2);

    Point pt_3(8.49, 0., 0.);
    auto nearest_neighbors_3 = FindkNearestNeighbors(points, pt_3, 2);
    std::vector<Point> expected_3 = {Point(8., 0., 0.), Point(9., 0., 0.)};
    EXPECT_EQ(nearest_neighbors_3, expected_3);

    Point pt_4(19.5, 0., 0.);
    auto nearest_neighbors_4 = FindkNearestNeighbors(points, pt_4, 2);
    std::vector<Point> expected_4 = {Point(9., 0., 0.), Point(8., 0., 0.)};
    EXPECT_EQ(nearest_neighbors_4, expected_4);
}

TEST(MatrixInterpolationTest, LinearlyInterpolate1x1Matrices) {
    auto m1 = Kokkos::View<double**>("m1", 1, 1);
    auto m1_host = Kokkos::create_mirror(m1);
    m1_host(0, 0) = 1.;
    Kokkos::deep_copy(m1, m1_host);

    auto m2 = Kokkos::View<double**>("m2", 1, 1);
    auto m2_host = Kokkos::create_mirror(m2);
    m2_host(0, 0) = 2.;
    Kokkos::deep_copy(m2, m2_host);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.), {{1.}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.5), {{1.5}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.25), {{1.25}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.75), {{1.75}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 1.), {{2.}}
    );
}

TEST(MatrixInterpolationTest, LinearlyInterpolate6x6Matrices) {
    auto m1 = Kokkos::View<double**>("m1", 6, 6);
    auto m1_host = Kokkos::create_mirror(m1);
    m1_host(0, 0) = 1.;
    m1_host(1, 1) = 2.;
    m1_host(2, 2) = 3.;
    m1_host(3, 3) = 4.;
    m1_host(4, 4) = 5.;
    m1_host(5, 5) = 6.;
    Kokkos::deep_copy(m1, m1_host);

    auto m2 = Kokkos::View<double**>("m2", 6, 6);
    auto m2_host = Kokkos::create_mirror(m2);
    m2_host(0, 0) = 7.;
    m2_host(1, 1) = 8.;
    m2_host(2, 2) = 9.;
    m2_host(3, 3) = 10.;
    m2_host(4, 4) = 11.;
    m2_host(5, 5) = 12.;
    Kokkos::deep_copy(m2, m2_host);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.),
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 2., 0., 0., 0., 0.},  // row 2
            {0., 0., 3., 0., 0., 0.},  // row 3
            {0., 0., 0., 4., 0., 0.},  // row 4
            {0., 0., 0., 0., 5., 0.},  // row 5
            {0., 0., 0., 0., 0., 6.}   // row 6
        }
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.25),
        {
            {2.5, 0., 0., 0., 0., 0.},  // row 1
            {0., 3.5, 0., 0., 0., 0.},  // row 2
            {0., 0., 4.5, 0., 0., 0.},  // row 3
            {0., 0., 0., 5.5, 0., 0.},  // row 4
            {0., 0., 0., 0., 6.5, 0.},  // row 5
            {0., 0., 0., 0., 0., 7.5}   // row 6
        }
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.5),
        {
            {4., 0., 0., 0., 0., 0.},  // row 1
            {0., 5., 0., 0., 0., 0.},  // row 2
            {0., 0., 6., 0., 0., 0.},  // row 3
            {0., 0., 0., 7., 0., 0.},  // row 4
            {0., 0., 0., 0., 8., 0.},  // row 5
            {0., 0., 0., 0., 0., 9.}   // row 6
        }
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 0.75),
        {
            {5.5, 0., 0., 0., 0., 0.},  // row 1
            {0., 6.5, 0., 0., 0., 0.},  // row 2
            {0., 0., 7.5, 0., 0., 0.},  // row 3
            {0., 0., 0., 8.5, 0., 0.},  // row 4
            {0., 0., 0., 0., 9.5, 0.},  // row 5
            {0., 0., 0., 0., 0., 10.5}  // row 6
        }
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearlyInterpolateMatrices(m1, m2, 1.),
        {
            {7., 0., 0., 0., 0., 0.},   // row 1
            {0., 8., 0., 0., 0., 0.},   // row 2
            {0., 0., 9., 0., 0., 0.},   // row 3
            {0., 0., 0., 10., 0., 0.},  // row 4
            {0., 0., 0., 0., 11., 0.},  // row 5
            {0., 0., 0., 0., 0., 12.}   // row 6
        }
    );
}

TEST(ShapeFunctionsTest, ZerothOrderLegendrePolynomial) {
    EXPECT_EQ(LegendrePolynomial(0, -1.), 1.);
    EXPECT_EQ(LegendrePolynomial(0, 0.), 1.);
    EXPECT_EQ(LegendrePolynomial(0, 1.), 1.);
}

TEST(ShapeFunctionsTest, FirstOrderLegendrePolynomial) {
    EXPECT_EQ(LegendrePolynomial(1, -1.), -1.);
    EXPECT_EQ(LegendrePolynomial(1, 0.), 0.);
    EXPECT_EQ(LegendrePolynomial(1, 1.), 1.);
}

TEST(ShapeFunctionsTest, SecondOrderLegendrePolynomial) {
    EXPECT_EQ(LegendrePolynomial(2, -1.), 1.);
    EXPECT_EQ(LegendrePolynomial(2, 0.), -0.5);
    EXPECT_EQ(LegendrePolynomial(2, 1.), 1.);
}

TEST(ShapeFunctionsTest, ThirdOrderLegendrePolynomial) {
    EXPECT_EQ(LegendrePolynomial(3, -1.), -1.);
    EXPECT_EQ(LegendrePolynomial(3, 0.), 0.);
    EXPECT_EQ(LegendrePolynomial(3, 1.), 1.);
}

TEST(ShapeFunctionsTest, FourthOrderLegendrePolynomial) {
    EXPECT_EQ(LegendrePolynomial(4, -1.), 1.);
    EXPECT_EQ(LegendrePolynomial(4, -0.6546536707079771), -0.4285714285714286);
    EXPECT_EQ(LegendrePolynomial(4, 0.), 0.375);
    EXPECT_EQ(LegendrePolynomial(4, 0.6546536707079771), -0.4285714285714286);
    EXPECT_EQ(LegendrePolynomial(4, 1.), 1.);
}

TEST(ShapeFunctionsTest, FindGLLPointsForFirstOrderPolynomial) {
    auto gll_points = GenerateGLLPoints(1);
    std::vector<double> expected = {-1., 1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, FindGLLPointsForSecondOrderPolynomial) {
    auto gll_points = GenerateGLLPoints(2);
    std::vector<double> expected = {-1., 0., 1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, FindGLLPointsForFourthOrderPolynomial) {
    auto gll_points = GenerateGLLPoints(4);
    std::vector<double> expected = {-1., -0.6546536707079771437983, 0., 0.654653670707977143798, 1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, FindGLLPointsForSixthOrderPolynomial) {
    auto gll_points = GenerateGLLPoints(6);
    std::vector<double> expected = {-1., -0.8302238962785669, -0.46884879347071423,
                                    0.,  0.46884879347071423, 0.8302238962785669,
                                    1.};

    EXPECT_EQ(gll_points.size(), expected.size());
    for (size_t i = 0; i < gll_points.size(); ++i) {
        EXPECT_NEAR(gll_points[i], expected[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, ZerothOrderLegendrePolynomialDerivative) {
    EXPECT_EQ(LegendrePolynomialDerivative(0, -1.), 0.);
    EXPECT_EQ(LegendrePolynomialDerivative(0, 0.), 0.);
    EXPECT_EQ(LegendrePolynomialDerivative(0, 1.), 0.);
}

TEST(ShapeFunctionsTest, FirstOrderLegendrePolynomialDerivative) {
    EXPECT_EQ(LegendrePolynomialDerivative(1, -1.), 1.);
    EXPECT_EQ(LegendrePolynomialDerivative(1, 0.), 1.);
    EXPECT_EQ(LegendrePolynomialDerivative(1, 1.), 1.);
}

TEST(ShapeFunctionsTest, SecondOrderLegendrePolynomialDerivative) {
    EXPECT_EQ(LegendrePolynomialDerivative(2, -1.), -3.);
    EXPECT_EQ(LegendrePolynomialDerivative(2, 0.), 0.);
    EXPECT_EQ(LegendrePolynomialDerivative(2, 1.), 3.);
}

TEST(ShapeFunctionsTest, ThirdOrderLegendrePolynomialDerivative) {
    EXPECT_EQ(LegendrePolynomialDerivative(3, -1.), 6.);
    EXPECT_EQ(LegendrePolynomialDerivative(3, 0.), -1.5);
    EXPECT_EQ(LegendrePolynomialDerivative(3, 1.), 6.);
}

TEST(ShapeFunctionsTest, SixthOrderLegendrePolynomialDerivative) {
    EXPECT_EQ(LegendrePolynomialDerivative(6, -1.), -21.);
    EXPECT_EQ(LegendrePolynomialDerivative(6, 0.), 0.);
    EXPECT_EQ(LegendrePolynomialDerivative(6, 1.), 21.);
}

TEST(ShapeFunctionsTest, FirstOrderLagrangePolynomial) {
    auto lagrange_poly_1 = LagrangePolynomial(1, -1.);
    std::vector<double> expected = {1., 0.};

    EXPECT_EQ(lagrange_poly_1.size(), expected.size());
    for (size_t i = 0; i < lagrange_poly_1.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_1[i], expected[i], 1e-15);
    }

    auto lagrange_poly_2 = LagrangePolynomial(1, 0.);
    std::vector<double> expected_2 = {0.5, 0.5};

    EXPECT_EQ(lagrange_poly_2.size(), expected_2.size());
    for (size_t i = 0; i < lagrange_poly_2.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_2[i], expected_2[i], 1e-15);
    }

    auto lagrange_poly_3 = LagrangePolynomial(1, 1.);
    std::vector<double> expected_3 = {0., 1.};

    EXPECT_EQ(lagrange_poly_3.size(), expected_3.size());
    for (size_t i = 0; i < lagrange_poly_3.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_3[i], expected_3[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, SecondOrderLagrangePolynomial) {
    auto lagrange_poly_1 = LagrangePolynomial(2, -1.);
    std::vector<double> expected = {1., 0., 0.};

    EXPECT_EQ(lagrange_poly_1.size(), expected.size());
    for (size_t i = 0; i < lagrange_poly_1.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_1[i], expected[i], 1e-15);
    }

    auto lagrange_poly_2 = LagrangePolynomial(2, 0.);
    std::vector<double> expected_2 = {0., 1., 0.};

    EXPECT_EQ(lagrange_poly_2.size(), expected_2.size());
    for (size_t i = 0; i < lagrange_poly_2.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_2[i], expected_2[i], 1e-15);
    }

    auto lagrange_poly_3 = LagrangePolynomial(2, 1.);
    std::vector<double> expected_3 = {0., 0., 1.};

    EXPECT_EQ(lagrange_poly_3.size(), expected_3.size());
    for (size_t i = 0; i < lagrange_poly_3.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_3[i], expected_3[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, FourthOrderLagrangePolynomial) {
    auto lagrange_poly_1 = LagrangePolynomial(4, -1.);
    std::vector<double> expected = {1., 0., 0., 0., 0.};

    EXPECT_EQ(lagrange_poly_1.size(), expected.size());
    for (size_t i = 0; i < lagrange_poly_1.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_1[i], expected[i], 1e-15);
    }

    auto lagrange_poly_2 = LagrangePolynomial(4, -0.6546536707079771);
    std::vector<double> expected_2 = {0., 1., 0., 0., 0.};

    EXPECT_EQ(lagrange_poly_2.size(), expected_2.size());
    for (size_t i = 0; i < lagrange_poly_2.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_2[i], expected_2[i], 1e-15);
    }

    auto lagrange_poly_3 = LagrangePolynomial(4, 0.);
    std::vector<double> expected_3 = {0., 0., 1., 0., 0.};

    EXPECT_EQ(lagrange_poly_3.size(), expected_3.size());
    for (size_t i = 0; i < lagrange_poly_3.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_3[i], expected_3[i], 1e-15);
    }

    auto lagrange_poly_4 = LagrangePolynomial(4, 0.6546536707079771);
    std::vector<double> expected_4 = {0., 0., 0., 1., 0.};

    EXPECT_EQ(lagrange_poly_4.size(), expected_4.size());
    for (size_t i = 0; i < lagrange_poly_4.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_4[i], expected_4[i], 1e-15);
    }

    auto lagrange_poly_5 = LagrangePolynomial(4, 1.);
    std::vector<double> expected_5 = {0., 0., 0., 0., 1.};

    EXPECT_EQ(lagrange_poly_5.size(), expected_5.size());
    for (size_t i = 0; i < lagrange_poly_5.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_5[i], expected_5[i], 1e-15);
    }

    auto lagrange_poly_6 = LagrangePolynomial(4, -0.8);
    std::vector<double> expected_6 = {
        0.266400000000000, 0.855336358376291, -0.1776000000000001, 0.0854636416237095,
        -0.0296000000000000};

    EXPECT_EQ(lagrange_poly_6.size(), expected_6.size());
    for (size_t i = 0; i < lagrange_poly_6.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_6[i], expected_6[i], 1e-15);
    }

    auto lagrange_poly_7 = LagrangePolynomial(4, 0.1);
    std::vector<double> expected_7 = {
        0.0329625, -0.1121093731918499, 0.9669, 0.1525343731918499, -0.0402875};

    EXPECT_EQ(lagrange_poly_7.size(), expected_7.size());
    for (size_t i = 0; i < lagrange_poly_7.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_7[i], expected_7[i], 1e-15);
    }

    auto lagrange_poly_8 = LagrangePolynomial(4, 0.4);
    std::vector<double> expected_8 = {
        0.0564, -0.1746924181056723, 0.5263999999999998, 0.7234924181056726, -0.1316};

    EXPECT_EQ(lagrange_poly_8.size(), expected_8.size());
    for (size_t i = 0; i < lagrange_poly_8.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_8[i], expected_8[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, FirstOrderLagrangePolynomialDerivative) {
    auto lagrange_poly_der_1 = LagrangePolynomialDerivative(1, -1.);
    std::vector<double> expected = {-0.5, 0.5};

    EXPECT_EQ(lagrange_poly_der_1.size(), expected.size());
    for (size_t i = 0; i < lagrange_poly_der_1.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_1[i], expected[i], 1e-15);
    }

    auto lagrange_poly_der_2 = LagrangePolynomialDerivative(1, 1.);
    std::vector<double> expected_2 = {-0.5, 0.5};

    EXPECT_EQ(lagrange_poly_der_2.size(), expected_2.size());
    for (size_t i = 0; i < lagrange_poly_der_2.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_2[i], expected_2[i], 1e-15);
    }
}

TEST(ShapeFunctionsTest, FourthOrderLagrangePolynomialDerivative) {
    auto lagrange_poly_der_1 = LagrangePolynomialDerivative(4, -1.);
    std::vector<double> expected = {
        -5., 6.75650248872424, -2.666666666666667, 1.410164177942427, -0.5};

    EXPECT_EQ(lagrange_poly_der_1.size(), expected.size());
    for (size_t i = 0; i < lagrange_poly_der_1.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_1[i], expected[i], 1e-15);
    }

    auto lagrange_poly_der_2 = LagrangePolynomialDerivative(4, -0.6546536707079771);
    std::vector<double> expected_2 = {
        -1.240990253030983, 0., 1.74574312188794, -0.7637626158259736, 0.2590097469690172};

    EXPECT_EQ(lagrange_poly_der_2.size(), expected_2.size());
    for (size_t i = 0; i < lagrange_poly_der_2.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_2[i], expected_2[i], 1e-15);
    }

    auto lagrange_poly_der_3 = LagrangePolynomialDerivative(4, 0.);
    std::vector<double> expected_3 = {0.375, -1.336584577695454, 0., 1.336584577695454, -0.375};

    EXPECT_EQ(lagrange_poly_der_3.size(), expected_3.size());
    for (size_t i = 0; i < lagrange_poly_der_3.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_3[i], expected_3[i], 1e-15);
    }

    auto lagrange_poly_der_4 = LagrangePolynomialDerivative(4, 0.6546536707079771);
    std::vector<double> expected_4 = {
        -0.2590097469690172, 0.7637626158259736, -1.74574312188794, 0., 1.240990253030983};

    EXPECT_EQ(lagrange_poly_der_4.size(), expected_4.size());
    for (size_t i = 0; i < lagrange_poly_der_4.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_4[i], expected_4[i], 1e-15);
    }

    auto lagrange_poly_der_5 = LagrangePolynomialDerivative(4, 1.);
    std::vector<double> expected_5 = {
        0.5, -1.410164177942427, 2.666666666666667, -6.756502488724241, 5.};

    EXPECT_EQ(lagrange_poly_der_5.size(), expected_5.size());
    for (size_t i = 0; i < lagrange_poly_der_5.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_5[i], expected_5[i], 1e-15);
    }

    auto lagrange_poly_der_6 = LagrangePolynomialDerivative(4, -0.8);
    std::vector<double> expected_6 = {
        -2.497, 2.144324478146484, 0.5546666666666656, -0.31499114481315, 0.1129999999999999};

    EXPECT_EQ(lagrange_poly_der_6.size(), expected_6.size());
    for (size_t i = 0; i < lagrange_poly_der_6.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_6[i], expected_6[i], 1e-15);
    }

    auto lagrange_poly_der_7 = LagrangePolynomialDerivative(4, 0.1);
    std::vector<double> expected_7 = {
        0.27725, -0.896320373697923, -0.6573333333333338, 1.696653707031257, -0.42025};

    EXPECT_EQ(lagrange_poly_der_7.size(), expected_7.size());
    for (size_t i = 0; i < lagrange_poly_der_7.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_7[i], expected_7[i], 1e-15);
    }

    auto lagrange_poly_der_8 = LagrangePolynomialDerivative(4, 0.4);
    std::vector<double> expected_8 = {
        -0.1210000000000001, 0.4156426862650312, -2.069333333333333, 1.805690647068303,
        -0.03099999999999978};

    EXPECT_EQ(lagrange_poly_der_8.size(), expected_8.size());
    for (size_t i = 0; i < lagrange_poly_der_8.size(); ++i) {
        EXPECT_NEAR(lagrange_poly_der_8[i], expected_8[i], 1e-15);
    }
}

}  // namespace openturbine::gebt_poc::tests
