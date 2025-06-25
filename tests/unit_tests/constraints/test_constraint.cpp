#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "constraints/constraint.hpp"

namespace openturbine::tests {

class ConstraintTest : public ::testing::Test {
protected:
    const size_t id{1};        // constraint id
    const size_t node1_id{1};  // base node id
    const size_t node2_id{2};  // target node id
    const std::array<size_t, 2> node_ids{node1_id, node2_id};
    const Array_3 axis_vec{0., 1., 0.};                                 // axis vector is y-axis
    const Array_7 init_displacement{1., 2., 3., 0.707, 0.707, 0., 0.};  // reference displacement
};

TEST_F(ConstraintTest, DefaultInitialization) {
    const auto constraint = Constraint(id, ConstraintType::kFixedBC, node_ids);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kFixedBC);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, (Array_3{0., 0., 0.}));
    EXPECT_EQ(constraint.initial_displacement, (Array_7{0., 0., 0., 1., 0., 0., 0.}));
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, FixedBCInitialization) {
    const auto constraint = Constraint(id, ConstraintType::kFixedBC, node_ids);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kFixedBC);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, (Array_3{0., 0., 0.}));
    EXPECT_EQ(constraint.initial_displacement, (Array_7{0., 0., 0., 1., 0., 0., 0.}));
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, RevoluteJointInitialization) {
    const auto constraint = Constraint(id, ConstraintType::kRevoluteJoint, node_ids, axis_vec);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kRevoluteJoint);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, axis_vec);
    EXPECT_EQ(constraint.initial_displacement, (Array_7{0., 0., 0., 1., 0., 0., 0.}));
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, PrescribedBCInitialization) {
    const auto constraint =
        Constraint(id, ConstraintType::kPrescribedBC, node_ids, axis_vec, init_displacement);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kPrescribedBC);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, axis_vec);
    EXPECT_EQ(constraint.initial_displacement, init_displacement);
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, RotationControlInitialization) {
    double control_signal = 0.;
    const auto constraint = Constraint(
        id, ConstraintType::kRotationControl, node_ids, axis_vec,
        Array_7{0., 0., 0., 1., 0., 0., 0.}, &control_signal
    );

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kRotationControl);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, axis_vec);
    EXPECT_EQ(constraint.initial_displacement, (Array_7{0., 0., 0., 1., 0., 0., 0.}));
    EXPECT_EQ(constraint.control, &control_signal);
}

TEST_F(ConstraintTest, RigidJointInitialization) {
    const auto constraint = Constraint(id, ConstraintType::kRigidJoint, node_ids, axis_vec);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kRigidJoint);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, axis_vec);
    EXPECT_EQ(constraint.initial_displacement, (Array_7{0., 0., 0., 1., 0., 0., 0.}));
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, FixedBC3DOFsInitialization) {
    const auto constraint = Constraint(id, ConstraintType::kFixedBC3DOFs, node_ids, axis_vec);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kFixedBC3DOFs);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, axis_vec);
    EXPECT_EQ(constraint.initial_displacement, (Array_7{0., 0., 0., 1., 0., 0., 0.}));
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, PrescribedBC3DOFsInitialization) {
    const auto constraint =
        Constraint(id, ConstraintType::kPrescribedBC3DOFs, node_ids, axis_vec, init_displacement);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kPrescribedBC3DOFs);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, axis_vec);
    EXPECT_EQ(constraint.initial_displacement, init_displacement);
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, RigidJoint6DOFsTo3DOFsInitialization) {
    const auto constraint =
        Constraint(id, ConstraintType::kRigidJoint6DOFsTo3DOFs, node_ids, axis_vec);

    EXPECT_EQ(constraint.ID, id);
    EXPECT_EQ(constraint.type, ConstraintType::kRigidJoint6DOFsTo3DOFs);
    EXPECT_EQ(constraint.node_ids[0], node1_id);
    EXPECT_EQ(constraint.node_ids[1], node2_id);
    EXPECT_EQ(constraint.axis_vector, axis_vec);
    EXPECT_EQ(constraint.initial_displacement, (Array_7{0., 0., 0., 1., 0., 0., 0.}));
    EXPECT_EQ(constraint.control, nullptr);
}

TEST_F(ConstraintTest, CustomInitialDisplacement) {
    const Array_7 custom_disp{1.5, 2.5, 3.5, 0.866, 0.5, 0., 0.};
    const auto constraint =
        Constraint(id, ConstraintType::kPrescribedBC, node_ids, axis_vec, custom_disp);

    EXPECT_EQ(constraint.initial_displacement, custom_disp);
}

TEST_F(ConstraintTest, NullControlSignal) {
    const auto constraint = Constraint(
        id, ConstraintType::kRotationControl, node_ids, axis_vec,
        Array_7{0., 0., 0., 1., 0., 0., 0.}, nullptr
    );

    EXPECT_EQ(constraint.control, nullptr);
}

}  // namespace openturbine::tests
