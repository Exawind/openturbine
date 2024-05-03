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

    BeamElement(
        std::vector<BeamNode> nodes_, std::vector<BeamSection> sections_, BeamQuadrature quadrature_
    )
        : nodes(std::move(nodes_)),
          sections(std::move(sections_)),
          quadrature(std::move(quadrature_)) {}
};

}  // namespace openturbine
