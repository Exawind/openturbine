#include <iostream>

#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/restruct_poc/beams/beam_element.hpp"
#include "src/restruct_poc/model/model.hpp"

namespace openturbine::restruct_poc::tests {

class ModelFixture : public ::testing::Test {
protected:
    void SetUp() override {
        pos = {0., 0., 0.};
        rot = {1., 0., 0., 0.};
        v = {0., 0., 0.};
        omega = {0., 0., 0.};
    }

    Array_3 pos;
    Array_4 rot;
    Array_3 v;
    Array_3 omega;

    Model model;
};

TEST_F(ModelFixture, AddNodeToModel) {
    ASSERT_EQ(model.NumNodes(), 0);

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
}

TEST_F(ModelFixture, TranslateModelNode) {
    // Add a node to the model and check the ID
    auto node = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );

    // Get the node and check the position
    auto node_0 = model.GetNode(0);
    ASSERT_EQ(node_0->ID, 0);
    ASSERT_EQ(node_0->x[0], pos[0]);  // 0.
    ASSERT_EQ(node_0->x[1], pos[1]);  // 0.
    ASSERT_EQ(node_0->x[2], pos[2]);  // 0.

    // Now translate the node and check the new position
    Array_3 displacement = {1., 2., 3.};
    node_0->Translate(displacement);
    ASSERT_EQ(node_0->x[0], pos[0] + displacement[0]);  // 1.
    ASSERT_EQ(node_0->x[1], pos[1] + displacement[1]);  // 2.
    ASSERT_EQ(node_0->x[2], pos[2] + displacement[2]);  // 3.
}

TEST_F(ModelFixture, RotateModelNode) {
    // Add a node to the model and check the ID
    auto node = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );

    // Translate the node to {1., 0., 0.}
    auto node_0 = model.GetNode(0);
    node_0->Translate({1., 0., 0.});

    // Now rotate the node 90 degrees around the z-axis
    node_0->Rotate({0., 0., 1.}, M_PI / 2.);
    ASSERT_NEAR(node_0->x[0], 0., 1e-12);
    ASSERT_NEAR(node_0->x[1], 1., 1e-12);
    ASSERT_NEAR(node_0->x[2], 0., 1e-12);

    // Return the node to {1., 0., 0.}
    node_0->Translate({1., -1., 0.});

    // Now rotate the node 45 degrees around the z-axis using a quaternion
    node_0->Rotate({0.92388, 0., 0., 0.382683});
    ASSERT_NEAR(node_0->x[0], 0.707107, 1e-6);
    ASSERT_NEAR(node_0->x[1], 0.707107, 1e-6);
    ASSERT_NEAR(node_0->x[2], 0., 1e-6);
}

TEST_F(ModelFixture, AddBeamElementToModel) {
    // Add couple of nodes to the model
    auto node1 = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );
    auto node2 = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );

    // Add a beam element to the model
    auto nodes = std::vector<BeamNode>{BeamNode{0., *node1}, BeamNode{1., *node2}};
    auto sections = std::vector<BeamSection>{};
    auto quadrature = BeamQuadrature{};
    auto beam_element = model.AddBeamElement(nodes, sections, quadrature);

    // Get number of elements in the model
    ASSERT_EQ(model.NumBeamElements(), 1);

    // Get the elements in the model and check their number
    auto elements = model.GetBeamElements();
    ASSERT_EQ(elements.size(), 1);
}

}  // namespace openturbine::restruct_poc::tests
