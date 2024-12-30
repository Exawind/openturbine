#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/constraints/constraint_type.hpp"

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

TEST(ConstraintTypeTest, RevoluteJointHasFiveDOFs) {
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kRevoluteJoint), 5U);
}

TEST(ConstraintTypeTest, NoneConstraintHasSixDOFs) {
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kNone), 6U);
}

TEST(ConstraintTypeTest, FixedBCHasSixDOFs) {
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kFixedBC), 6U);
}

TEST(ConstraintTypeTest, PrescribedBCHasSixDOFs) {
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kPrescribedBC), 6U);
}

TEST(ConstraintTypeTest, RigidJointHasSixDOFs) {
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kRigidJoint), 6U);
}

TEST(ConstraintTypeTest, RotationControlHasSixDOFs) {
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kRotationControl), 6U);
}

TEST(ConstraintTypeTest, InvalidConstraintTypeThrowsError) {
    EXPECT_THROW(NumDOFsForConstraint(static_cast<ConstraintType>(100)), std::runtime_error);
}

}  // namespace openturbine::tests
