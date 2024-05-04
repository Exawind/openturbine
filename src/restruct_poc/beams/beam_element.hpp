#pragma once

#include <array>
#include <vector>

#include "beam_node.hpp"
#include "beam_section.hpp"

namespace openturbine {

struct BeamElement {
    using BeamQuadrature = std::vector<std::array<double, 2>>;

    std::vector<BeamNode> nodes;        // Element node positions/rotations in material frame
    std::vector<BeamSection> sections;  // Element mass/stiffness in material frame
    BeamQuadrature quadrature;          // Element quadrature points and weights

    BeamElement(std::vector<BeamNode> n, std::vector<BeamSection> s, BeamQuadrature q)
        : nodes(std::move(n)), sections(std::move(s)), quadrature(std::move(q)) {}
};

}  // namespace openturbine
