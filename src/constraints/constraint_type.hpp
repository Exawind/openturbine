#pragma once

#include <stdexcept>

namespace openturbine {

enum class ConstraintType : std::uint8_t {
    kNone = 0,             //< No constraint -- default type
    kFixedBC = 1,          //< Fixed boundary condition/clamped constraint -- all DOFs fixed at
                           //< the node where the constraint is applied
    kPrescribedBC = 2,     //< Prescribed boundary condition -- displacement and orientation values
                           //< are specified => all DOFs are defined at the node
    kRigidJoint = 3,       //< Rigid constraint between two nodes -- no relative motion permitted
                           //< between the nodes => all DOFs of target node are constrained
    kRevoluteJoint = 4,    //< Target node rotates freely around a specified axis --
                           //< all but one DOFs are constrained
    kRotationControl = 5,  //< A rotation is specified about a given axis and other DOFs
                           //< are constrained => all DOFs are constrained/specified
};

/// Returns the number of nodes used/required by the constraint type
KOKKOS_INLINE_FUNCTION
constexpr size_t GetNumberOfNodes(ConstraintType t) {
    // Rigid joint/revolute joint/rotation control constraints require two nodes
    const auto has_two_nodes = t == ConstraintType::kRigidJoint ||
                               t == ConstraintType::kRevoluteJoint ||
                               t == ConstraintType::kRotationControl;

    // Default is one node (fixed and prescribed BCs)
    return 1U + static_cast<size_t>(has_two_nodes);
}

/// Returns the number of degrees of freedom prescribed/fixed by the constraint type
KOKKOS_INLINE_FUNCTION
constexpr size_t NumDOFsForConstraint(ConstraintType type) {
    // A revolute joint constraint fixes 5 DOFs
    if (type == ConstraintType::kRevoluteJoint) {
        return 5U;
    }

    // All other constraints fix 6 DOFs
    if (type == ConstraintType::kNone || type == ConstraintType::kFixedBC ||
        type == ConstraintType::kPrescribedBC || type == ConstraintType::kRigidJoint ||
        type == ConstraintType::kRotationControl) {
        return 6U;
    }

    throw std::runtime_error("Invalid constraint type");
}

}  // namespace openturbine
