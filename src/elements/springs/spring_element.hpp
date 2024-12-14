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

    SpringElement(std::array<Node, 2> nodes, double k, double l0 = 0.)
        : nodes(nodes), stiffness(k), undeformed_length(l0) {
        // If undeformed length is not provided, compute it from the nodes
        if (l0 == 0.) {
            const auto& p1 = nodes[0].x;
            const auto& p2 = nodes[1].x;
            undeformed_length = std::sqrt(
                std::pow(p2[0] - p1[0], 2) + std::pow(p2[1] - p1[1], 2) + std::pow(p2[2] - p1[2], 2)
            );
        }
    }
};

}  // namespace openturbine
