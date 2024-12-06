#pragma once

namespace openturbine {

/**
 * @brief Enumeration to define the type of the constraint
 */
enum class ConstraintType : std::uint8_t {
    kNone = 0,           //< No constraint (default)
    kFixedBC = 1,        //< Fixed boundary condition/clamped constraint (all 6 DOFs fixed)
    kPrescribedBC = 2,   //< Prescribed boundary condition (displacement and rotation are specified
                         // i.e. all 6 DOFs are fixed)
    kRigidJoint = 3,     // Rigid constraint between two nodes (no relative motion between nodes i.e.
                         // all 6 DOFs of target node constrained)
    kRevoluteJoint = 4,  //< Target node rotates freely around a specified axis (5 DOFs constrained)
    kRotationControl = 5,  //< A rotation is specified about a given axis (1 DOF allowed about the
                           // axis specified by the user - the other 5 DOFs are constrained)
};

/**
 * @brief Returns the number of nodes used by the constraint type
 *
 * @param t Constraint type
 * @return Number of nodes
 */
KOKKOS_INLINE_FUNCTION
constexpr size_t GetNumberOfNodes(ConstraintType t) {
    // Rigid joint, revolute joint, and rotation control have two nodes
    const auto has_two_nodes = t == ConstraintType::kRigidJoint ||
                               t == ConstraintType::kRevoluteJoint ||
                               t == ConstraintType::kRotationControl;
    // Default is one node (fixed and prescribed BCs)
    return 1U + static_cast<size_t>(has_two_nodes);
}

/**
 * @brief Returns the number of degrees of freedom used/fixed by the constraint type
 *
 * @param type Constraint type
 * @return Number of degrees of freedom
 */
KOKKOS_INLINE_FUNCTION
constexpr size_t NumDOFsForConstraint(ConstraintType type) {
    switch (type) {
        case ConstraintType::kRevoluteJoint: {
            return 5U;  // A revolute joint constraint fixes 5 DOFs
        } break;
        default:
            return static_cast<size_t>(6U);  // Default: Fixes 6 DOFs
    }
}

}  // namespace openturbine
