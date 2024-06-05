#pragma once

#include <vector>

#include "mass_element.hpp"
#include "masses.hpp"

namespace openturbine {

template <typename T1, typename T2>
inline void PopulateElementViews(const MassElement& elem, T1 node_x0, T2 node_Mstar) {
    // Copy initial position/rotation
    for (size_t k = 0; k < kLieGroupComponents; ++k) {
        node_x0(k) = elem.node.initial_dofs[k];
    }

    // Copy mass matrix
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            node_Mstar(i, j) = elem.M_star[i][j];
        }
    }
}

}  // namespace openturbine
