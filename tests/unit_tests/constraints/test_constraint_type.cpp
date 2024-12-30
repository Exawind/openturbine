#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/constraints/constraint_type.hpp"

namespace openturbine::tests {

TEST(ConstraintTypeTest, GetNumberOfNodesReturnsCorrectCount) {
    // Single node constraints
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kNone), 1U);
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kFixedBC), 1U);
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kPrescribedBC), 1U);

    // Two node constraints
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kRigidJoint), 2U);
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kRevoluteJoint), 2U);
    EXPECT_EQ(GetNumberOfNodes(ConstraintType::kRotationControl), 2U);
}

TEST(ConstraintTypeTest, NumDOFsForConstraintReturnsCorrectCount) {
    // Special case: Revolute joint has 5 DOFs
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kRevoluteJoint), 5U);

    // All other constraints have 6 DOFs
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kNone), 6U);
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kFixedBC), 6U);
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kPrescribedBC), 6U);
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kRigidJoint), 6U);
    EXPECT_EQ(NumDOFsForConstraint(ConstraintType::kRotationControl), 6U);

    // Expect a runtime error for an invalid constraint type
    EXPECT_THROW(NumDOFsForConstraint(static_cast<ConstraintType>(100)), std::runtime_error);
}

}  // namespace openturbine::tests
