#pragma once

#include "src/model/node.hpp"
#include "src/types.hpp"

namespace openturbine {

/**
 * @brief Mass element consists of only one node and a single section defined by a mass matrix
 */
struct MassElement {
    Node node;                                    // Mass element has a single node
    std::array<std::array<double, 6>, 6> M_star;  // Mass matrix in material frame

    MassElement(Node n, std::array<std::array<double, 6>, 6> mass_matrix)
        : node(n), M_star(mass_matrix) {}
};

}  // namespace openturbine
