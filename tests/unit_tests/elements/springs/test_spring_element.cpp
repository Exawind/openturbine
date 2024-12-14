#include <gtest/gtest.h>

#include "src/elements/springs/spring_element.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

TEST(SpringElement, CreateSpringWithSpecifiedLength) {
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
    constexpr double l0 = 2.;   // undeformed length
    SpringElement spring({*node1, *node2}, k, l0);

    EXPECT_EQ(spring.stiffness, k);
    EXPECT_EQ(spring.undeformed_length, l0);
    EXPECT_EQ(spring.nodes[0].x[0], 0.0);
    EXPECT_EQ(spring.nodes[1].x[0], 1.0);
}

TEST(SpringElement, CreateSpringWithComputedLength) {
    Model model;

    // Create two nodes with 3 units of distance between them
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

    // Create spring with only stiffness specified (length should be computed)
    constexpr double k = 100.;  // stiffness
    SpringElement spring({*node1, *node2}, k);

    EXPECT_EQ(spring.stiffness, k);
    EXPECT_NEAR(spring.undeformed_length, 3., 1e-12);
    EXPECT_EQ(spring.nodes[0].x[0], 0.0);
    EXPECT_EQ(spring.nodes[1].x[0], 2.0);
}

}  // namespace openturbine::tests
