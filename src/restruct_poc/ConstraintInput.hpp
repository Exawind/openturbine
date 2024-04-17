#pragma once

namespace openturbine {

struct ConstraintInput {
    int base_node_index;
    int constrained_node_index;
    ConstraintInput(int node1, int node2) : base_node_index(node1), constrained_node_index(node2) {}
};

}  // namespace openturbine
