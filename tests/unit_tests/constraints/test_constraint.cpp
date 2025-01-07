#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/constraints/constraint.hpp"

namespace openturbine::tests {

class ConstraintTest : public ::testing::Test {
protected:
    const size_t id{1};                                        // constraint id
    const Node node1{1, Array_7{1., 0., 0., 1., 0., 0., 0.}};  // base node
    const Node node2{2, Array_7{2., 0., 0., 1., 0., 0., 0.}};  // target node
    const Array_3 ref_vec{0., 1., 0.};                         // reference vector is y-axis
};

TEST_F(ConstraintTest, FixedBCInitialization) {
    const auto constraint = Constraint(ConstraintType::kFixedBC, id, node1, node2);

    EXPECT_EQ(constraint.type, ConstraintType::kFixedBC);
    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.base_node.x, node1.x);
    EXPECT_EQ(constraint.target_node.x, node2.x);

    // For fixed BC, X0 should be at the target node position
    const Array_3 expected_X0{2., 0., 0.};
    EXPECT_EQ(constraint.X0, expected_X0);
}

TEST_F(ConstraintTest, RevoluteJointInitialization) {
    const auto constraint = Constraint(ConstraintType::kRevoluteJoint, id, node1, node2, ref_vec);

    EXPECT_EQ(constraint.type, ConstraintType::kRevoluteJoint);

    // For revolute joint, X0 should be the relative position between the nodes
    const Array_3 expected_X0{1., 0., 0.};
    EXPECT_EQ(constraint.X0, expected_X0);

    // For revolute joint, axes should form an orthonormal basis
    EXPECT_NEAR(Norm(constraint.x_axis), 1., 1e-10);
    EXPECT_NEAR(Norm(constraint.y_axis), 1., 1e-10);
    EXPECT_NEAR(Norm(constraint.z_axis), 1., 1e-10);
    EXPECT_NEAR(DotProduct(constraint.x_axis, constraint.y_axis), 0., 1e-10);
    EXPECT_NEAR(DotProduct(constraint.y_axis, constraint.z_axis), 0., 1e-10);
    EXPECT_NEAR(DotProduct(constraint.z_axis, constraint.x_axis), 0., 1e-10);
}

TEST_F(ConstraintTest, RotationControlInitialization) {
    const auto constraint = Constraint(ConstraintType::kRotationControl, id, node1, node2, ref_vec);

    EXPECT_EQ(constraint.type, ConstraintType::kRotationControl);

    // For rotation control, X0 should be the relative position between the nodes
    const Array_3 expected_X0{1., 0., 0.};
    EXPECT_EQ(constraint.X0, expected_X0);

    // For rotation control, x_axis should be unit vector of reference vector
    const Array_3 expected_x_axis = UnitVector(ref_vec);
    EXPECT_EQ(constraint.x_axis, expected_x_axis);
}

TEST_F(ConstraintTest, DefaultInitialization) {
    const auto constraint = Constraint(ConstraintType::kNone, id, node1, node2);

    // X0 should be relative position between nodes
    const Array_3 expected_X0{1., 0., 0.};
    EXPECT_EQ(constraint.X0, expected_X0);
}

}  // namespace openturbine::tests
