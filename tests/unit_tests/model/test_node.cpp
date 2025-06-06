#include <iostream>

#include <gtest/gtest.h>

#include "model/model.hpp"

namespace openturbine::tests {

TEST(NodeTest, DisplacedPosition_TranslationOnly) {
    Model model;

    // Create node with initial position and translational displacement only
    constexpr auto init_position = std::array{1., 2., 3.};
    constexpr auto init_orientation = std::array{1., 0., 0., 0.};
    constexpr auto displacement = std::array{2., -1., 0.5};
    auto node_id =
        model.AddNode()
            .SetPosition(
                init_position[0], init_position[1], init_position[2], init_orientation[0],
                init_orientation[1], init_orientation[2], init_orientation[3]
            )
            .SetDisplacement(displacement[0], displacement[1], displacement[2], 1., 0., 0., 0.)
            .Build();

    auto node = model.GetNode(node_id);
    auto displaced_position = node.DisplacedPosition();

    // Translation should be added to initial position
    ASSERT_NEAR(displaced_position[0], init_position[0] + displacement[0], 1e-12);  // 3.
    ASSERT_NEAR(displaced_position[1], init_position[1] + displacement[1], 1e-12);  // 1.
    ASSERT_NEAR(displaced_position[2], init_position[2] + displacement[2], 1e-12);  // 3.5
    // Orientation should remain unchanged
    ASSERT_NEAR(displaced_position[3], init_orientation[0], 1e-12);
    ASSERT_NEAR(displaced_position[4], init_orientation[1], 1e-12);
    ASSERT_NEAR(displaced_position[5], init_orientation[2], 1e-12);
    ASSERT_NEAR(displaced_position[6], init_orientation[3], 1e-12);
}

TEST(NodeTest, DisplacedPosition_RotationOnly) {
    Model model;

    // Create node with initial position and rotational displacement only
    constexpr auto init_position = std::array{1., 2., 3.};
    constexpr auto init_orientation = std::array{1., 0., 0., 0.};  // identity quaternion
    // 90° rotation around z-axis
    const auto rotation = RotationVectorToQuaternion({0., 0., M_PI / 2.});
    auto node_id =
        model.AddNode()
            .SetPosition(
                init_position[0], init_position[1], init_position[2], init_orientation[0],
                init_orientation[1], init_orientation[2], init_orientation[3]
            )
            .SetDisplacement(0., 0., 0., rotation[0], rotation[1], rotation[2], rotation[3])
            .Build();

    auto node = model.GetNode(node_id);
    auto displaced_position = node.DisplacedPosition();

    // Position should remain unchanged
    ASSERT_NEAR(displaced_position[0], init_position[0], 1e-12);
    ASSERT_NEAR(displaced_position[1], init_position[1], 1e-12);
    ASSERT_NEAR(displaced_position[2], init_position[2], 1e-12);
    // Orientation should be the displacement rotation (90° around z)
    ASSERT_NEAR(displaced_position[3], rotation[0], 1e-12);
    ASSERT_NEAR(displaced_position[4], rotation[1], 1e-12);
    ASSERT_NEAR(displaced_position[5], rotation[2], 1e-12);
    ASSERT_NEAR(displaced_position[6], rotation[3], 1e-12);
}

TEST(NodeTest, DisplacedPosition_TranslationAndRotation) {
    Model model;

    // Create node with both translational and rotational displacement
    constexpr auto init_position = std::array{1., 0., 0.};
    constexpr auto init_orientation = std::array{1., 0., 0., 0.};
    constexpr auto disp_position = std::array{1., 2., 3.};
    const auto disp_rotation = RotationVectorToQuaternion({0., M_PI / 3., 0.});  // 60° around y axis

    auto node_id = model.AddNode()
                       .SetPosition(
                           init_position[0], init_position[1], init_position[2], init_orientation[0],
                           init_orientation[1], init_orientation[2], init_orientation[3]
                       )
                       .SetDisplacement(
                           disp_position[0], disp_position[1], disp_position[2], disp_rotation[0],
                           disp_rotation[1], disp_rotation[2], disp_rotation[3]
                       )
                       .Build();

    auto node = model.GetNode(node_id);
    auto displaced_position = node.DisplacedPosition();

    // Check translation is added correctly
    ASSERT_NEAR(displaced_position[0], init_position[0] + disp_position[0], 1e-12);  // 2.
    ASSERT_NEAR(displaced_position[1], init_position[1] + disp_position[1], 1e-12);  // 2.
    ASSERT_NEAR(displaced_position[2], init_position[2] + disp_position[2], 1e-12);  // 3.

    // Check rotation composition
    auto expected_orientation = QuaternionCompose(init_orientation, disp_rotation);
    ASSERT_NEAR(displaced_position[3], expected_orientation[0], 1e-12);
    ASSERT_NEAR(displaced_position[4], expected_orientation[1], 1e-12);
    ASSERT_NEAR(displaced_position[5], expected_orientation[2], 1e-12);
    ASSERT_NEAR(displaced_position[6], expected_orientation[3], 1e-12);
}

TEST(NodeTest, Translate) {
    Model model;

    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};

    // Add a node to the model and check the initial position
    auto node_id = model.AddNode()
                       .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                       .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                       .Build();

    auto node_0 = model.GetNode(node_id);
    ASSERT_EQ(node_0.id, 0);
    ASSERT_EQ(node_0.x0[0], pos[0]);  // 0.
    ASSERT_EQ(node_0.x0[1], pos[1]);  // 0.
    ASSERT_EQ(node_0.x0[2], pos[2]);  // 0.

    // Now translate the node and check the new position
    Array_3 displacement = {1., 2., 3.};
    node_0.Translate(displacement);
    ASSERT_EQ(node_0.x0[0], pos[0] + displacement[0]);  // 1.
    ASSERT_EQ(node_0.x0[1], pos[1] + displacement[1]);  // 2.
    ASSERT_EQ(node_0.x0[2], pos[2] + displacement[2]);  // 3.
}

TEST(NodeTest, RotateAboutPoint) {
    Model model;

    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};

    auto node_id = model.AddNode()
                       .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                       .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                       .Build();

    // Translate the node to {1., 0., 0.}
    auto node_0 = model.GetNode(node_id);
    node_0.Translate({1., 0., 0.});

    // Now rotate the node 90 degrees around the z-axis
    node_0.RotateAboutPoint(Array_3{0., 0., M_PI / 2.}, {0., 0., 0.});
    ASSERT_NEAR(node_0.x0[0], 0., 1e-12);
    ASSERT_NEAR(node_0.x0[1], 1., 1e-12);
    ASSERT_NEAR(node_0.x0[2], 0., 1e-12);

    // Return the node to {1., 0., 0.}
    node_0.Translate({1., -1., 0.});

    // Now rotate the node 45 degrees around the z-axis using a quaternion
    node_0.RotateAboutPoint({0.92388, 0., 0., 0.382683}, {0., 0., 0.});
    ASSERT_NEAR(node_0.x0[0], 0.707107, 1e-6);
    ASSERT_NEAR(node_0.x0[1], 0.707107, 1e-6);
    ASSERT_NEAR(node_0.x0[2], 0., 1e-6);
}

TEST(NodeTest, TranslateDisplacement) {
    Model model;

    // Create node with zero displacement and unit orientation
    constexpr auto u_pos = std::array{0., 0., 0.};
    constexpr auto u_rot = std::array{1., 0., 0., 0.};
    auto node_id =
        model.AddNode()
            .SetDisplacement(u_pos[0], u_pos[1], u_pos[2], u_rot[0], u_rot[1], u_rot[2], u_rot[3])
            .Build();

    // Add some displacement to the node and check the new displacement
    Array_3 displacement_1 = {1., 2., 3.};
    auto node_0 = model.GetNode(node_id);
    node_0.TranslateDisplacement(displacement_1);
    ASSERT_EQ(node_0.u[0], u_pos[0] + displacement_1[0]);  // 1.
    ASSERT_EQ(node_0.u[1], u_pos[1] + displacement_1[1]);  // 2.
    ASSERT_EQ(node_0.u[2], u_pos[2] + displacement_1[2]);  // 3.

    // Add another displacement and check the cumulative effect
    Array_3 displacement_2 = {2., 1., 0.};
    node_0.TranslateDisplacement(displacement_2);
    ASSERT_EQ(node_0.u[0], u_pos[0] + displacement_1[0] + displacement_2[0]);  // 3.
    ASSERT_EQ(node_0.u[1], u_pos[1] + displacement_1[1] + displacement_2[1]);  // 3.
    ASSERT_EQ(node_0.u[2], u_pos[2] + displacement_1[2] + displacement_2[2]);  // 3.
}

TEST(NodeTest, RotateDisplacementAboutPoint) {
    Model model;

    // Create node with initial displacement and unit orientation
    constexpr auto u_pos = std::array{1., 0., 0.};
    constexpr auto u_rot = std::array{1., 0., 0., 0.};
    auto node_id =
        model.AddNode()
            .SetDisplacement(u_pos[0], u_pos[1], u_pos[2], u_rot[0], u_rot[1], u_rot[2], u_rot[3])
            .Build();

    // Rotate displacement 90 degrees around z-axis about origin
    auto node_0 = model.GetNode(node_id);
    node_0.RotateDisplacementAboutPoint(Array_3{0., 0., M_PI / 2.}, {0., 0., 0.});
    // Check that the displacement is now (0, 1, 0)
    ASSERT_NEAR(node_0.u[0], 0., 1e-12);
    ASSERT_NEAR(node_0.u[1], 1., 1e-12);
    ASSERT_NEAR(node_0.u[2], 0., 1e-12);
    // Check that the orientation is now (√2/2, 0, 0, √2/2)
    ASSERT_NEAR(node_0.u[3], 0.707107, 1e-6);
    ASSERT_NEAR(node_0.u[4], 0., 1e-12);
    ASSERT_NEAR(node_0.u[5], 0., 1e-12);
    ASSERT_NEAR(node_0.u[6], 0.707107, 1e-6);

    // Return displacement to (1, 0, 0)
    node_0.TranslateDisplacement({1., -1., 0.});
    // Return orientation to initial state (1, 0, 0, 0)
    node_0.u[3] = 1.;
    node_0.u[4] = 0.;
    node_0.u[5] = 0.;
    node_0.u[6] = 0.;

    // Rotate displacement 45 degrees around z-axis about origin
    node_0.RotateDisplacementAboutPoint({0.92388, 0., 0., 0.382683}, {0., 0., 0.});
    // Check that the displacement is now (0.707107, 0.707107, 0)
    ASSERT_NEAR(node_0.u[0], 0.707107, 1e-6);
    ASSERT_NEAR(node_0.u[1], 0.707107, 1e-6);
    ASSERT_NEAR(node_0.u[2], 0., 1e-12);
    // Check that the orientation is now (0.924, 0, 0, 0.383)
    ASSERT_NEAR(node_0.u[3], 0.92388, 1e-6);
    ASSERT_NEAR(node_0.u[4], 0., 1e-12);
    ASSERT_NEAR(node_0.u[5], 0., 1e-12);
    ASSERT_NEAR(node_0.u[6], 0.382683, 1e-6);
}

}  // namespace openturbine::tests
