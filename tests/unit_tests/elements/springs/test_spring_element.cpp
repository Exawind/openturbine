#include <gtest/gtest.h>

#include "src/elements/springs/spring_element.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

TEST(SpringElement, CreateSpringElement_ZeroUndeformedLength) {
    Model model;

    // Create two nodes
    auto node1 = model.AddNode(
        {0., 0., 0., 1., 0., 0., 0.},  // position
        {0., 0., 0., 1., 0., 0., 0.},  // displacement
        {0., 0., 0., 0., 0., 0.}       // velocity
    );
    auto node2 = model.AddNode(
        {1., 0., 0., 1., 0., 0., 0.},  // position
        {0., 0., 0., 1., 0., 0., 0.},  // displacement
        {0., 0., 0., 0., 0., 0.}       // velocity
    );

    constexpr double k = 100.;  // stiffness
    constexpr double l0 = 0.;   // undeformed length
    SpringElement spring({*node1, *node2}, k, l0);

    EXPECT_EQ(spring.stiffness, k);
    EXPECT_EQ(spring.undeformed_length, l0);
    EXPECT_EQ(spring.nodes[0].x[0], 0.);
    EXPECT_EQ(spring.nodes[1].x[0], 1.);
}

TEST(SpringElement, CreateSpringElement_NonZeroUndeformedLength) {
    Model model;

    // Create two nodes
    auto node1 = model.AddNode(
        {0., 0., 0., 1., 0., 0., 0.},  // position
        {0., 0., 0., 1., 0., 0., 0.},  // displacement
        {0., 0., 0., 0., 0., 0.}       // velocity
    );
    auto node2 = model.AddNode(
        {2., 2., 1., 1., 0., 0., 0.},  // position
        {0., 0., 0., 1., 0., 0., 0.},  // displacement
        {0., 0., 0., 0., 0., 0.}       // velocity
    );

    constexpr double k = 100.;  // stiffness
    constexpr double l0 = 3.;   // undeformed length
    SpringElement spring({*node1, *node2}, k, l0);

    EXPECT_EQ(spring.stiffness, k);
    EXPECT_EQ(spring.undeformed_length, l0);
    EXPECT_EQ(spring.nodes[0].x[0], 0.);
    EXPECT_EQ(spring.nodes[1].x[0], 2.);
}

}  // namespace openturbine::tests
