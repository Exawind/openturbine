#pragma once

#include <array>
#include <vector>

#include "beam_node.hpp"
#include "beam_section.hpp"

#include "src/types.hpp"

namespace openturbine {

/**
 * @brief Beam element constitutes flexible beams material behavior in openturbine.
 *
 * @details A beam element is defined by a set of nodes and sections. Each section is defined by a
 * 6x6 mass matrix and a 6x6 stiffness matrix. The element quadrature is used to integrate the
 * mass and stiffness matrices along the length of the beam.
 */
struct BeamElement {
    std::vector<BeamNode> nodes;        // Element node positions/rotations in material frame
    std::vector<BeamSection> sections;  // Element mass/stiffness in material frame
    BeamQuadrature quadrature;          // Element quadrature points and weights

    BeamElement(std::vector<BeamNode> n, std::vector<BeamSection> s, BeamQuadrature q)
        : nodes(std::move(n)), sections(std::move(s)), quadrature(std::move(q)) {}
};

}  // namespace openturbine
