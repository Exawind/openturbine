#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "constraints/constraint_type.hpp"

namespace openturbine::tests {

TEST(ConstraintTypeTest, NoneConstraintHasOneNode) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kNone), 1U);
}

TEST(ConstraintTypeTest, FixedBCHasOneNode) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kFixedBC), 1U);
}

TEST(ConstraintTypeTest, PrescribedBCHasOneNode) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kPrescribedBC), 1U);
}

TEST(ConstraintTypeTest, RigidJointHasTwoNodes) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kRigidJoint), 2U);
}

TEST(ConstraintTypeTest, RevoluteJointHasTwoNodes) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kRevoluteJoint), 2U);
}

TEST(ConstraintTypeTest, RotationControlHasTwoNodes) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kRotationControl), 2U);
}

TEST(ConstraintTypeTest, FixedBC3DOFsHasOneNode) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kFixedBC3DOFs), 1U);
}

TEST(ConstraintTypeTest, PrescribedBC3DOFsHasOneNode) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kPrescribedBC3DOFs), 1U);
}

TEST(ConstraintTypeTest, RigidJoint6DOFsTo3DOFsHasTwoNodes) {
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kRigidJoint6DOFsTo3DOFs), 2U);
}

TEST(ConstraintTypeTest, RevoluteJointHasFiveDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kRevoluteJoint), 5U);
}

TEST(ConstraintTypeTest, NoneConstraintHasSixDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kNone), 6U);
}

TEST(ConstraintTypeTest, FixedBCHasSixDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kFixedBC), 6U);
}

TEST(ConstraintTypeTest, PrescribedBCHasSixDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kPrescribedBC), 6U);
}

TEST(ConstraintTypeTest, RigidJointHasSixDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kRigidJoint), 6U);
}

TEST(ConstraintTypeTest, RotationControlHasSixDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kRotationControl), 6U);
}

TEST(ConstraintTypeTest, FixedBC3DOFsHasThreeDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kFixedBC3DOFs), 3U);
}

TEST(ConstraintTypeTest, PrescribedBC3DOFsHasThreeDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kPrescribedBC3DOFs), 3U);
}

TEST(ConstraintTypeTest, RigidJoint6DOFsTo3DOFsHasThreeDOFs) {
    EXPECT_EQ(NumRowsForConstraint(ConstraintType::kRigidJoint6DOFsTo3DOFs), 3U);
}

}  // namespace openturbine::tests
