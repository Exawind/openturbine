#pragma once

#include "src/model/node.hpp"

namespace openturbine {

struct BeamNode {
    double position;  // Position of node in element on range [0, 1]
    Node node;

    BeamNode(const Node& n) : position(0.), node(n) {}
    BeamNode(double s, const Node& n) : position(s), node(n) {}
};

}  // namespace openturbine
