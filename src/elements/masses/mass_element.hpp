#pragma once

#include "src/model/node.hpp"

namespace openturbine {

/**
 * @brief Mass element constitutes rigid bodies/masses material behavior in openturbine.
 * It has a single node and a single section completely defined by a 6x6 mass matrix.
 */
struct MassElement {
    Node node;                                    // 1 node
    std::array<std::array<double, 6>, 6> M_star;  // Mass matrix in material frame

    MassElement(Node n, std::array<std::array<double, 6>, 6> mass_matrix)
        : node(std::move(n)), M_star(std::move(mass_matrix)) {}
};

}  // namespace openturbine
