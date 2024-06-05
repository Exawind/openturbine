#pragma once

namespace openturbine {

enum ConstraintType {
    Rigid,
    FixedBC,
    PrescribedBC,
    Cylindrical,
    RotationControl,
};

struct ConstraintInput {
    int base_node_index;
    int target_node_index;
    int type;
    std::array<double, 3> x0;

    ConstraintInput(int node1, int node2, ConstraintType constraint_type)
        : base_node_index(node1),
          target_node_index(node2),
          type(constraint_type),
          x0({0., 0., 0.}) {}

    static ConstraintInput RigidConstraint(int node1, int node2) {
        return ConstraintInput(node1, node2, ConstraintType::Rigid);
    }
    static ConstraintInput PrescribedBC(
        int node, std::array<double, 3> ref_position = {0., 0., 0.}
    ) {
        auto ci = ConstraintInput(-1, node, ConstraintType::PrescribedBC);
        ci.x0 = ref_position;
        return ci;
    }
    static ConstraintInput FixedBC(int node) {
        return ConstraintInput(-1, node, ConstraintType::FixedBC);
    }
    static ConstraintInput Cylindrical(int node1, int node2) {
        return ConstraintInput(node1, node2, ConstraintType::Cylindrical);
    }
    static ConstraintInput RotationControl(int node1, int node2) {
        return ConstraintInput(node1, node2, ConstraintType::RotationControl);
    }
};

}  // namespace openturbine
