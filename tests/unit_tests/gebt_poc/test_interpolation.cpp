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
    Kokkos::View<double**> m1("m1", 1, 1);
    Kokkos::View<double**> m2("m2", 1, 1);
    m1(0, 0) = 1.;
    m2(0, 0) = 2.;

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearInterpolation(m1, m2, 0.), {{1.}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearInterpolation(m1, m2, 0.5), {{1.5}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearInterpolation(m1, m2, 0.25), {{1.25}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearInterpolation(m1, m2, 0.75), {{1.75}}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearInterpolation(m1, m2, 1.), {{2.}}
    );
}

TEST(MatrixInterpolationTest, LinearlyInterpolate6x6Matrices) {
    Kokkos::View<double**> m1("m1", 6, 6);
    Kokkos::deep_copy(m1, 0.);
    m1(0, 0) = 1.;
    m1(1, 1) = 2.;
    m1(2, 2) = 3.;
    m1(3, 3) = 4.;
    m1(4, 4) = 5.;
    m1(5, 5) = 6.;

    Kokkos::View<double**> m2("m2", 6, 6);
    Kokkos::deep_copy(m2, 0.);
    m2(0, 0) = 7.;
    m2(1, 1) = 8.;
    m2(2, 2) = 9.;
    m2(3, 3) = 10.;
    m2(4, 4) = 11.;
    m2(5, 5) = 12.;

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        LinearInterpolation(m1, m2, 0.),
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
        LinearInterpolation(m1, m2, 0.25),
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
        LinearInterpolation(m1, m2, 0.5),
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
        LinearInterpolation(m1, m2, 0.75),
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
        LinearInterpolation(m1, m2, 1.),
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

}  // namespace openturbine::gebt_poc::tests
