#pragma once

#include <array>
#include <cmath>

#include "src/model/node.hpp"

namespace openturbine {

/**
 * @brief Spring element represents a constitutively linear spring connecting two nodes and defined
 * by its scalar stiffness and undeformed length.
 */
struct SpringElement {
    std::array<Node, 2> nodes;  // 2 nodes (start and end points of spring)
    double stiffness;           // Spring stiffness coefficient
    double undeformed_length;   // Reference/undeformed length of spring

    SpringElement(std::array<Node, 2> n, double k, double l0)
        : nodes(n), stiffness(k), undeformed_length(l0) {}
};

}  // namespace openturbine
