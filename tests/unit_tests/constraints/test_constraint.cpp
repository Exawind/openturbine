#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/constraints/constraint.hpp"

namespace openturbine::tests {

class ConstraintTest : public ::testing::Test {
protected:
    const size_t id{1};        // constraint id
    const size_t node1_id{1};  // base node id
    const size_t node2_id{2};  // target node id
    const std::array<size_t, 2> node_ids{node1_id, node2_id};
    const Array_3 ref_vec{0., 1., 0.};  // reference vector is y-axis
};

TEST_F(ConstraintTest, FixedBCInitialization) {
    const auto constraint = Constraint(id, ConstraintType::kFixedBC, node_ids);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kFixedBC);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.node_num_dofs, (std::array<size_t, 2>{6U, 6U}));  // Default DOFs
}

TEST_F(ConstraintTest, RevoluteJointInitialization) {
    const auto constraint =
        Constraint(id, ConstraintType::kRevoluteJoint, node_ids, {6U, 6U}, ref_vec);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kRevoluteJoint);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.node_num_dofs, (std::array<size_t, 2>{6U, 6U}));
    EXPECT_EQ(constraint.vec, ref_vec);
}

TEST_F(ConstraintTest, RotationControlInitialization) {
    double control_signal = 0.;
    const auto constraint = Constraint(
        id, ConstraintType::kRotationControl, node_ids, {6U, 3U}, ref_vec, &control_signal
    );

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kRotationControl);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.node_num_dofs, (std::array<size_t, 2>{6U, 3U}));
    EXPECT_EQ(constraint.vec, ref_vec);
    EXPECT_EQ(constraint.control, &control_signal);
}

}  // namespace openturbine::tests
