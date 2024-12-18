#include <iostream>

#include <gtest/gtest.h>

#include "src/elements/beams/beam_element.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

TEST(Model, AddNodeToModel) {
    Model model;
    ASSERT_EQ(model.NumNodes(), 0);

    // Add a node to the model
    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};
    auto node = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );

    ASSERT_EQ(node->ID, 0);
    ASSERT_EQ(model.NumNodes(), 1);

    auto nodes = model.GetNodes();
    ASSERT_EQ(nodes.size(), 1);
}

TEST(Model, TranslateModelNode) {
    Model model;

    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};

    // Add a node to the model and check the initial position
    auto node = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );

    auto node_0 = model.GetNode(0);
    ASSERT_EQ(node_0.ID, 0);
    ASSERT_EQ(node_0.x[0], pos[0]);  // 0.
    ASSERT_EQ(node_0.x[1], pos[1]);  // 0.
    ASSERT_EQ(node_0.x[2], pos[2]);  // 0.

    // Now translate the node and check the new position
    Array_3 displacement = {1., 2., 3.};
    node_0.Translate(displacement);
    ASSERT_EQ(node_0.x[0], pos[0] + displacement[0]);  // 1.
    ASSERT_EQ(node_0.x[1], pos[1] + displacement[1]);  // 2.
    ASSERT_EQ(node_0.x[2], pos[2] + displacement[2]);  // 3.
}

TEST(Model, RotateModelNode) {
    Model model;

    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};

    auto node = model.AddNode(
        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
    );

    // Translate the node to {1., 0., 0.}
    auto node_0 = model.GetNode(0);
    node_0.Translate({1., 0., 0.});

    // Now rotate the node 90 degrees around the z-axis
    node_0.Rotate({0., 0., 1.}, M_PI / 2.);
    ASSERT_NEAR(node_0.x[0], 0., 1e-12);
    ASSERT_NEAR(node_0.x[1], 1., 1e-12);
    ASSERT_NEAR(node_0.x[2], 0., 1e-12);

    // Return the node to {1., 0., 0.}
    node_0.Translate({1., -1., 0.});

    // Now rotate the node 45 degrees around the z-axis using a quaternion
    node_0.Rotate({0.92388, 0., 0., 0.382683});
    ASSERT_NEAR(node_0.x[0], 0.707107, 1e-6);
    ASSERT_NEAR(node_0.x[1], 0.707107, 1e-6);
    ASSERT_NEAR(node_0.x[2], 0., 1e-6);
}

TEST(Model, AddBeamElementToModel) {
    Model model;

    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};

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

    ASSERT_EQ(model.NumBeamElements(), 1);

    auto elements = model.GetBeamElements();
    ASSERT_EQ(elements.size(), 1);
}

TEST(Model, AddMassElementToModel) {
    Model model;
    auto node = model.AddNode(
        {0., 0., 0., 1., 0., 0., 0.},  // position
        {0., 0., 0., 1., 0., 0., 0.},  // displacement
        {0., 0., 0., 0., 0., 0.}       // velocity
    );
    auto mass_matrix = std::array<std::array<double, 6>, 6>{};
    auto mass_element = model.AddMassElement(*node, mass_matrix);

    ASSERT_EQ(model.NumMassElements(), 1);

    auto elements = model.GetMassElements();
    ASSERT_EQ(elements.size(), 1);
}

TEST(Model, ModelConstructorWithDefaults) {
    const Model model;
    ASSERT_EQ(model.NumNodes(), 0);
    ASSERT_EQ(model.NumBeamElements(), 0);
    ASSERT_EQ(model.NumMassElements(), 0);
    ASSERT_EQ(model.NumConstraints(), 0);
}

TEST(Model, ModelConstructorWithObjects) {
    auto nodes = std::vector<Node>{Node{0, {0., 0., 0.}, {0., 0., 0., 1., 0., 0., 0.}}};  // 1 node
    auto beam_elements = std::vector<BeamElement>{};  // 0 beam elements
    auto mass_elements = std::vector<MassElement>{};  // 0 mass elements
    auto constraints = std::vector<Constraint>{};     // 0 constraints

    const Model model(nodes, beam_elements, mass_elements, constraints);
    ASSERT_EQ(model.NumNodes(), 1);
    ASSERT_EQ(model.NumBeamElements(), 0);
    ASSERT_EQ(model.NumMassElements(), 0);
    ASSERT_EQ(model.NumConstraints(), 0);
}

TEST(Model, ModelConstructorWithPointers) {
    auto nodes = std::vector<std::shared_ptr<Node>>{};
    nodes.push_back(std::make_shared<Node>(
        0, std::array{0., 0., 0., 0., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0., 0.}
    ));                                                                      // 1 node
    const auto beam_elements = std::vector<std::shared_ptr<BeamElement>>{};  // 0 beam elements
    const auto mass_elements = std::vector<std::shared_ptr<MassElement>>{};  // 0 mass elements
    const auto constraints = std::vector<std::shared_ptr<Constraint>>{};     // 0 constraints

    const Model model(nodes, beam_elements, mass_elements, constraints);
    ASSERT_EQ(model.NumNodes(), 1);
    ASSERT_EQ(model.NumBeamElements(), 0);
    ASSERT_EQ(model.NumMassElements(), 0);
    ASSERT_EQ(model.NumConstraints(), 0);
}

TEST(Model, ModelCreateState) {
    Model model;

    // Rotation of 1 radian around x
    auto R1 = RotationVectorToQuaternion({1., 0., 0.});
    auto R2 = RotationVectorToQuaternion({0., 1., 0.});

    // Create node with initial position and displacement from initial position
    model.AddNode(
        std::array{1., 2., 3., R1[0], R1[1], R1[2], R1[3]},  // initial position/orientation
        std::array{3., 2., 1., R2[0], R2[1], R2[2], R2[3]}   // displacement
    );

    // Create state object from model
    auto state = model.CreateState();

    // Verify initial position
    const auto x0 = Kokkos::create_mirror(state.x0);
    Kokkos::deep_copy(x0, state.x0);
    const auto exact_x0 = std::array{1., 2., 3., R1[0], R1[1], R1[2], R1[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(x0(0, i), exact_x0[i], 1.e-15);
    }

    // Verify initial displacement
    const auto q = Kokkos::create_mirror(state.q);
    Kokkos::deep_copy(q, state.q);
    const auto exact_q = std::array{3., 2., 1., R2[0], R2[1], R2[2], R2[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(q(0, i), exact_q[i], 1.e-15);
    }

    // Verify current position (initial position plus displacement)
    auto Rt = QuaternionCompose(R2, R1);
    const auto x = Kokkos::create_mirror(state.x);
    Kokkos::deep_copy(x, state.x);
    const auto exact_x = std::array{4., 4., 4., Rt[0], Rt[1], Rt[2], Rt[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(x(0, i), exact_x[i], 1.e-15);
    }
}

}  // namespace openturbine::tests
