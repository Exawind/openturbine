#pragma once

#include "node.hpp"

#include "src/restruct_poc/solver/constraint.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

// InvalidNode represents an invalid node in constraints that only use the target node.
static Node InvalidNode(-1, {0., 0., 0., 1., 0., 0., 0.});

/// @brief Struct to define a turbine model with nodes and constraints
/// @details A model is a collection of nodes and constraints that define the geometry and
/// relationships between components in a turbine. Nodes represent points in space with
/// position, displacement, velocity, and acceleration. Constraints represent relationships
/// between nodes that restrict their relative motion in some way.
struct Model {
    std::vector<Node> nodes;
    std::vector<Constraint> constraints;

    /// Adds a node to the model and returns the node
    Node AddNode(
        const Array_7& position, const Array_7& displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        const Array_6& velocity = Array_6{0., 0., 0., 0., 0., 0.},
        const Array_6& acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    ) {
        this->nodes.push_back(
            Node(static_cast<int>(nodes.size()), position, displacement, velocity, acceleration)
        );
        return this->nodes.back();
    }

    /// Adds a rigid constraint to the model and returns the constraint.
    Constraint AddRigidConstraint(const Node& node1, const Node& node2) {
        this->constraints.push_back(Constraint(
            ConstraintType::kRigid, static_cast<int>(this->constraints.size()), node1, node2
        ));
        return this->constraints.back();
    }

    /// Adds a prescribed boundary condition constraint to the model and returns the constraint.
    Constraint AddPrescribedBC(const Node& node, const Array_3& ref_position = {0., 0., 0.}) {
        this->constraints.push_back(Constraint(
            ConstraintType::kPrescribedBC, static_cast<int>(this->constraints.size()), InvalidNode,
            node, ref_position
        ));
        return this->constraints.back();
    }

    /// Adds a fixed boundary condition constraint to the model and returns the constraint.
    Constraint AddFixedBC(const Node& node) {
        this->constraints.push_back(Constraint(
            ConstraintType::kFixedBC, static_cast<int>(this->constraints.size()), InvalidNode, node
        ));
        return this->constraints.back();
    }

    /// Adds a cylindrical constraint to the model and returns the constraint.
    Constraint AddCylindricalConstraint(const Node& node1, const Node& node2) {
        this->constraints.push_back(Constraint(
            ConstraintType::kCylindrical, static_cast<int>(this->constraints.size()), node1, node2
        ));
        return this->constraints.back();
    }

    /// Adds a rotation control constraint to the model and returns the constraint.
    Constraint AddRotationControl(
        const Node& node1, const Node& node2, const Array_3& axis, double* control
    ) {
        this->constraints.push_back(Constraint(
            ConstraintType::kRotationControl, static_cast<int>(this->constraints.size()), node1,
            node2, axis, control
        ));
        return this->constraints.back();
    }

    /// Returns the number of nodes in the model
    size_t NumNodes() { return this->nodes.size(); }
};

}  // namespace openturbine