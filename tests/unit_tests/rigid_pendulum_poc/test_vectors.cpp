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

}  // namespace openturbine::rigid_pendulum::tests
