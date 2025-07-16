#pragma once

#include <vector>

#include "beam_section.hpp"

namespace openturbine {

/**
 * @brief Beam element constitutes flexible beams material behavior in openturbine.
 *
 * @details A beam element is defined by a set of nodes and sections. Each section is defined by a
 * 6x6 mass matrix and a 6x6 stiffness matrix. The element quadrature is used to integrate the
 * mass and stiffness matrices along the length of the beam.
 */
struct BeamElement {
    size_t ID;                                      // Element identifier
    std::vector<size_t> node_ids;                   // Element node identifiers
    std::vector<BeamSection> sections;              // Element mass/stiffness in material frame
    std::vector<std::array<double, 2>> quadrature;  // Element quadrature points and weights

    BeamElement(
        size_t id, std::vector<size_t> n, std::vector<BeamSection> s,
        std::vector<std::array<double, 2>> q
    )
        : ID(id), node_ids(std::move(n)), sections(std::move(s)), quadrature(std::move(q)) {}
};

}  // namespace openturbine
