#pragma once

#include "node.hpp"

#include "src/restruct_poc/solver/constraint.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

static Node GroundNode(-1, {0., 0., 0., 1., 0., 0., 0.});

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

    const Constraint RigidConstraint(const Node node1, const Node node2) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), node1, node2, ConstraintType::Rigid)
        );
        return this->constraints.back();
    }
    const Constraint PrescribedBC(const Node node, Array_3 ref_position = {0., 0., 0.}) {
        this->constraints.push_back(Constraint(
            this->constraints.size(), GroundNode, node, ConstraintType::PrescribedBC, ref_position
        ));
        return this->constraints.back();
    }
    const Constraint FixedBC(const Node node) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), GroundNode, node, ConstraintType::FixedBC)
        );
        return this->constraints.back();
    }
    const Constraint Cylindrical(const Node node1, const Node node2) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), node1, node2, ConstraintType::Cylindrical)
        );
        return this->constraints.back();
    }
    const Constraint RotationControl(const Node node1, const Node node2) {
        this->constraints.push_back(
            Constraint(this->constraints.size(), node1, node2, ConstraintType::RotationControl)
        );
        return this->constraints.back();
    }

    int NumNodes() { return this->nodes.size(); }
};

}  // namespace openturbine