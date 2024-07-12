#pragma once

#include "node.hpp"

#include "src/restruct_poc/solver/constraint.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

// InvalidNode represents an invalid node in constraints that only use the target node.
static Node InvalidNode(-1, {0., 0., 0., 1., 0., 0., 0.});

struct Model {
    std::vector<Node> nodes;
    std::vector<Constraint> constraints;

    const Node AddNode(
        const Array_7& position, const Array_7& displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        const Array_6& velocity = Array_6{0., 0., 0., 0., 0., 0.},
        const Array_6& acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    ) {
        this->nodes.push_back(Node(nodes.size(), position, displacement, velocity, acceleration));
        return this->nodes.back();
    }

    const Constraint AddRigidConstraint(const Node& node1, const Node& node2) {
        this->constraints.push_back(
            Constraint(ConstraintType::kRigid, this->constraints.size(), node1, node2)
        );
        return this->constraints.back();
    }
    const Constraint AddPrescribedBC(const Node& node, const Array_3& ref_position = {0., 0., 0.}) {
        this->constraints.push_back(Constraint(
            ConstraintType::kPrescribedBC, this->constraints.size(), InvalidNode, node, ref_position
        ));
        return this->constraints.back();
    }
    const Constraint AddFixedBC(const Node& node) {
        this->constraints.push_back(
            Constraint(ConstraintType::kFixedBC, this->constraints.size(), InvalidNode, node)
        );
        return this->constraints.back();
    }
    const Constraint AddCylindricalConstraint(const Node& node1, const Node& node2) {
        this->constraints.push_back(
            Constraint(ConstraintType::kCylindrical, this->constraints.size(), node1, node2)
        );
        return this->constraints.back();
    }
    const Constraint AddRotationControl(
        const Node& node1, const Node& node2, const Array_3& axis, float* control
    ) {
        this->constraints.push_back(Constraint(
            ConstraintType::kRotationControl, this->constraints.size(), node1, node2, axis, control
        ));
        return this->constraints.back();
    }

    int NumNodes() { return this->nodes.size(); }
};

}  // namespace openturbine