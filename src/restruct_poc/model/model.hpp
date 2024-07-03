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
        Array_7 position, Array_7 displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        Array_6 velocity = Array_6{0., 0., 0., 0., 0., 0.},
        Array_6 acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    ) {
        this->nodes.push_back(Node(nodes.size(), position, displacement, velocity, acceleration));
        return this->nodes.back();
    }

    const Constraint AddRigidConstraint(const Node node1, const Node node2) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), node1, node2, ConstraintType::Rigid)
        );
        return this->constraints.back();
    }
    const Constraint AddPrescribedBC(const Node node, Array_3 ref_position = {0., 0., 0.}) {
        this->constraints.push_back(Constraint(
            this->constraints.size(), InvalidNode, node, ConstraintType::PrescribedBC, ref_position
        ));
        return this->constraints.back();
    }
    const Constraint AddFixedBC(const Node node) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), InvalidNode, node, ConstraintType::FixedBC)
        );
        return this->constraints.back();
    }
    const Constraint AddCylindricalConstraint(const Node node1, const Node node2) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), node1, node2, ConstraintType::Cylindrical)
        );
        return this->constraints.back();
    }
    const Constraint AddRotationControl(
        const Node node1, const Node node2, const Array_3 axis = {0., 0., 0.}
    ) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), node1, node2, ConstraintType::RotationControl, axis)
        );
        return this->constraints.back();
    }

    int NumNodes() { return this->nodes.size(); }
};

}  // namespace openturbine