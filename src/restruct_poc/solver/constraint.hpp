#pragma once

#include "src/restruct_poc/model/node.hpp"

namespace openturbine {

enum ConstraintType {
    Rigid,
    FixedBC,
    PrescribedBC,
    Cylindrical,
    RotationControl,
};

struct Constraint {
    int ID;
    Node base_node;
    Node target_node;
    ConstraintType type;
    Array_3 X0;  // reference position for prescribed BC

    Constraint(
        int id, const Node node1, const Node node2, ConstraintType constraint_type,
        Array_3 ref_position = {0., 0., 0.}
    )
        : ID(id), base_node(node1), target_node(node2), type(constraint_type) {
        switch (constraint_type) {
            case ConstraintType::FixedBC:
            case ConstraintType::PrescribedBC:
                X0[0] = this->target_node.x[0] - ref_position[0];
                X0[1] = this->target_node.x[1] - ref_position[1];
                X0[2] = this->target_node.x[2] - ref_position[2];
                break;
            default:
                X0[0] = this->target_node.x[0] - this->base_node.x[0];
                X0[1] = this->target_node.x[1] - this->base_node.x[1];
                X0[2] = this->target_node.x[2] - this->base_node.x[2];
        }
    }
};

}  // namespace openturbine
