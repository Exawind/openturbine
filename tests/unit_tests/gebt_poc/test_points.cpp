#include <gtest/gtest.h>

#include "src/gebt_poc/point.h"

namespace openturbine::gebt_poc::tests {

TEST(PointTest, DefaultConstructor) {
    Point point;

    EXPECT_EQ(point.GetXComponent(), 0.);
    EXPECT_EQ(point.GetYComponent(), 0.);
    EXPECT_EQ(point.GetZComponent(), 0.);
}

TEST(PointTest, ConstructorWithProvidedComponents) {
    Point point(1., 2., 3.);

    EXPECT_EQ(point.GetXComponent(), 1.);
    EXPECT_EQ(point.GetYComponent(), 2.);
    EXPECT_EQ(point.GetZComponent(), 3.);
}

TEST(PointTest, DistanceTo) {
    Point point1(1., 2., 3.);
    Point point2(4., 5., 6.);

    EXPECT_EQ(point1.DistanceTo(point2), 5.196152422706632);
}

TEST(PointTest, GetPositionVector) {
    Point point{1., 2., 3.};
    auto vector = point.GetPositionVector();
    std::tuple<double, double, double> expected = {1., 2., 3.};

    EXPECT_EQ(vector.GetComponents(), expected);
}

}  // namespace openturbine::gebt_poc::tests
