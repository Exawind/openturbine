#pragma once

#include "src/model/node.hpp"

namespace openturbine {

/**
 * @brief Mass element constitutes rigid bodies/masses material behavior in openturbine.
 * It has a single node and a single section completely defined by a 6x6 mass matrix.
 */
struct MassElement {
    size_t ID;                                    // Element identifier
    size_t node_id;                               // Node identifier
    std::array<std::array<double, 6>, 6> M_star;  // Mass matrix in material frame

    MassElement(size_t id, size_t n_id, std::array<std::array<double, 6>, 6> mass_matrix)
        : ID(id), node_id(n_id), M_star(mass_matrix) {}
};

}  // namespace openturbine
