#pragma once

#include <array>

#include "src/restruct_poc/model/node.hpp"

namespace openturbine {

struct BeamNode {
    double position;  // Position of node in element on range [0, 1]
    Node node;

    BeamNode(const Node n) : position(0.), node(n) {}
    BeamNode(double s, Node n) : position(s), node(n) {}
};

}  // namespace openturbine
