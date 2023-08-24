#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/vector.h"

namespace openturbine::rigid_pendulum::tests {

TEST(VectorTest, DefaultConstructor) {
    Vector v;
    std::tuple<double, double, double> expected = {0., 0., 0.};

    ASSERT_EQ(v.GetComponents(), expected);
}

TEST(VectorTest, ConstructorWithProvidedComponents) {
    Vector v(1., 2., 3.);
    std::tuple<double, double, double> expected = {1., 2., 3.};

    ASSERT_EQ(v.GetComponents(), expected);
}

TEST(VectorTest, GetIndividualComponents) {
    Vector v(1., 2., 3.);

    ASSERT_EQ(v.GetXComponent(), 1.);
    ASSERT_EQ(v.GetYComponent(), 2.);
    ASSERT_EQ(v.GetZComponent(), 3.);
}

TEST(VectorTest, Addition) {
    Vector v1(1., 2., 3.);
    Vector v2(4., 5., 6.);
    Vector v3 = v1 + v2;
    std::tuple<double, double, double> expected = {5., 7., 9.};

    ASSERT_EQ(v3.GetComponents(), expected);
}

TEST(VectorTest, Subtraction) {
    Vector v1(1., 2., 3.);
    Vector v2(4., 5., 6.);
    Vector v3 = v1 - v2;
    std::tuple<double, double, double> expected = {-3., -3., -3.};

    ASSERT_EQ(v3.GetComponents(), expected);
}

TEST(VectorTest, ScalarMultiplication) {
    Vector v1(1., 2., 3.);
    Vector v2 = v1 * 2.;
    std::tuple<double, double, double> expected = {2., 4., 6.};

    ASSERT_EQ(v2.GetComponents(), expected);
}

TEST(VectorTest, ScalarDivision) {
    Vector v1(1., 2., 3.);
    Vector v2 = v1 / 2.;
    std::tuple<double, double, double> expected = {0.5, 1., 1.5};

    ASSERT_EQ(v2.GetComponents(), expected);
}

TEST(VectorTest, Equality) {
    Vector v1(1., 2., 3.);
    Vector v2(1., 2., 3.);

    ASSERT_TRUE(v1 == v2);
}

TEST(VectorTest, Inequality) {
    Vector v1(0.33, -11.29, 22.12);
    Vector v2(0.33, -11.29, 22.13);

    ASSERT_FALSE(v1 == v2);
}

TEST(VectorTest, Length) {
    Vector v1(1., 2., 3.);
    double length = v1.Length();

    ASSERT_EQ(length, std::sqrt(1. * 1. + 2. * 2. + 3. * 3.));
}

TEST(VectorTest, ExpectNonUnitVector) {
    Vector v1(1., 1., 1.);

    ASSERT_FALSE(v1.IsUnitVector());
}

TEST(VectorTest, ExpectUnitVector) {
    Vector v1(1., 0., 0.);

    ASSERT_TRUE(v1.IsUnitVector());
}

TEST(VectorTest, ExpectNullVector) {
    Vector v1(0., 0., 0.);

    ASSERT_TRUE(v1.IsNullVector());
}

TEST(VectorTest, GetUnitVectorFromAProvidedVector) {
    Vector v1(1., 2., 3.);
    Vector v2 = v1.GetUnitVector();

    auto l = v1.Length();
    std::tuple<double, double, double> expected = {1. / l, 2. / l, 3. / l};

    ASSERT_EQ(v2.GetComponents(), expected);
}

TEST(VectorTest, DotProduct_Set1) {
    Vector v1(1., 2., 3.);
    Vector v2(4., 5., 6.);
    double dot_product = v1.DotProduct(v2);

    ASSERT_EQ(dot_product, 32.);
}

TEST(VectorTest, DotProduct_Set2) {
    Vector v1(-3.23, 17.19, 0.);
    Vector v2(0.37, -7.57, 1.11);
    double dot_product = v1.DotProduct(v2);

    ASSERT_EQ(dot_product, -3.23 * 0.37 + 17.19 * -7.57 + 0. * 1.11);
}

TEST(VectorTest, CrossProduct_Set1) {
    Vector v1(1., 2., 3.);
    Vector v2(4., 5., 6.);
    Vector v3 = v1.CrossProduct(v2);
    std::tuple<double, double, double> expected = {-3., 6., -3.};

    ASSERT_EQ(v3.GetComponents(), expected);
}

TEST(VectorTest, CrossProduct_Set2) {
    Vector v1(0.19, -5.03, 2.71);
    Vector v2(1.16, 0.09, 0.37);
    Vector v3 = v1.CrossProduct(v2);
    std::tuple<double, double, double> expected = {
        -5.03 * 0.37 - 2.71 * 0.09, 2.71 * 1.16 - 0.19 * 0.37, 0.19 * 0.09 - -5.03 * 1.16};

    ASSERT_EQ(v3.GetComponents(), expected);
}

TEST(VectorTest, ExpectNormalVectors) {
    Vector v1(1., 0., 0.);
    Vector v2(0., 1., 0.);

    ASSERT_TRUE(v1.IsNormalTo(v2));
}

TEST(VectorTest, ExpectNonNormalVectors) {
    Vector v1(1., 0., 0.);
    Vector v2(1., 1., 0.);

    ASSERT_FALSE(v1.IsNormalTo(v2));
}

TEST(VectorTest, ExpectParallelVectors) {
    Vector v1(1., 0., 0.);
    Vector v2(2., 0., 0.);

    ASSERT_TRUE(v1.IsParallelTo(v2));
}

TEST(VectorTest, ExpectNonParallelVectors) {
    Vector v1(1., 0., 0.);
    Vector v2(1., 1., 0.);

    ASSERT_FALSE(v1.IsParallelTo(v2));
}

}  // namespace openturbine::rigid_pendulum::tests
