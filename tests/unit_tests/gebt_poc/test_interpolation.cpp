#include <gtest/gtest.h>

#include "src/gebt_poc/interpolation.h"

namespace openturbine::gebt_poc::tests {

TEST(MatrixInterpolationTest, FindNearestNeighborOnALine) {
    // Create a list of points on a line from 0 to 10
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

}  // namespace openturbine::gebt_poc::tests
