#include <iostream>

#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/restruct_poc/model/model.hpp"

namespace openturbine::restruct_poc::tests {

// Write a test for the Model_2 class
TEST(Model_2, AddNode) {
    Model_2 model;
    Array_3 pos = {0., 0., 0.};
    Array_4 rot = {1., 0., 0., 0.};
    Array_3 v = {0., 0., 0.};
    Array_3 omega = {0., 0., 0.};

    // Add a node to the model and check the ID
    auto node = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );
    ASSERT_EQ(node->ID, 0);

    // Check the number of nodes in the model
    ASSERT_EQ(model.NumNodes(), 1);

    // Get the nodes in the model and check their number
    auto nodes = model.GetNodes();
    ASSERT_EQ(nodes.size(), 1);

    // Get the node and check the position
    auto node_0 = model.GetNode(0);
    ASSERT_EQ(node_0->ID, 0);

    // Check the position of the node
    ASSERT_EQ(node_0->x[0], pos[0]);  // 0.
    ASSERT_EQ(node_0->x[1], pos[1]);  // 0.
    ASSERT_EQ(node_0->x[2], pos[2]);  // 0.

    // Now translate the node and check the new position
    Array_3 displacement = {1., 2., 3.};
    node_0->Translate(displacement);
    ASSERT_EQ(node_0->x[0], pos[0] + displacement[0]);  // 1.
    ASSERT_EQ(node_0->x[1], pos[1] + displacement[1]);  // 2.
    ASSERT_EQ(node_0->x[2], pos[2] + displacement[2]);  // 3.

    // Add a second node to the model and check the ID
    auto node_1 = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );
    ASSERT_EQ(node_1->ID, 1);

    // Check the number of nodes in the model
    ASSERT_EQ(model.NumNodes(), 2);

    // Get the nodes in the model and check their number
    nodes = model.GetNodes();
    ASSERT_EQ(nodes.size(), 2);
}

}  // namespace openturbine::restruct_poc::tests
