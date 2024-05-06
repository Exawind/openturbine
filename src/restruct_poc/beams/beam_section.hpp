#pragma once

#include <array>

namespace openturbine {

struct BeamSection {
    double position;                              // Position of section in element on range [0, 1]
    std::array<std::array<double, 6>, 6> M_star;  // Mass matrix in material frame
    std::array<std::array<double, 6>, 6> C_star;  // Stiffness matrix in material frame

    BeamSection(
        double s, std::array<std::array<double, 6>, 6> mass,
        std::array<std::array<double, 6>, 6> stiffness
    )
        : position(s), M_star(std::move(mass)), C_star(std::move(stiffness)) {}
};

}  // namespace openturbine
