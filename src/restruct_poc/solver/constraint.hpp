#pragma once

#include "src/restruct_poc/model/node.hpp"

namespace openturbine {

enum class ConstraintType {
    None,
    FixedBC,       // Fixed boundary condition constraint (zero displacement)
    PrescribedBC,  // Prescribed boundary condition (displacement can be set)
    Rigid,  // Rigid constraint between two nodes (nodes maintain relative distance and rotation)
    Cylindrical,  // Target node rotates freely around specified axis. Relative distance and rotation
                  // are fixed)
    RotationControl,  // Specify rotation about given axis
};

struct Constraint {
    int ID;
    Node base_node;
    Node target_node;
    ConstraintType type;
    Array_3 X0;        // reference position for prescribed BC
    Array_3 rot_axis;  // unit vector axis for cylindrical constraint
    double* control;   // pointer to control variable

    Constraint(
        int id, const Node node1, const Node node2, ConstraintType constraint_type,
        Array_3 vec = {0., 0., 0.}, double* control_ = nullptr
    )
        : ID(id), base_node(node1), target_node(node2), type(constraint_type), control(control_) {
        switch (constraint_type) {
            case ConstraintType::FixedBC:
            case ConstraintType::PrescribedBC:
                X0[0] = this->target_node.x[0] - vec[0];
                X0[1] = this->target_node.x[1] - vec[1];
                X0[2] = this->target_node.x[2] - vec[2];
                break;
            case ConstraintType::RotationControl:
                rot_axis = vec;
                [[fallthrough]];
            default:
                X0[0] = this->target_node.x[0] - this->base_node.x[0];
                X0[1] = this->target_node.x[1] - this->base_node.x[1];
                X0[2] = this->target_node.x[2] - this->base_node.x[2];
        }
    }

    // NumDOFs returns the number of degrees of freedom used by constraint.
    int NumDOFs() const {
        switch (this->type) {
            case ConstraintType::Cylindrical:
                return 5;
            default:
                return kLieAlgebraComponents;
        }
    }
};

}  // namespace openturbine
