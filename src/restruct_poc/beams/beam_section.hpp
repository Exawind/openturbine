#pragma once

#include <array>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct BeamSection {
    double position;                              // Position of section in element on range [0, 1]
    std::array<std::array<double, 6>, 6> M_star;  // Mass matrix in material frame
    std::array<std::array<double, 6>, 6> C_star;  // Stiffness matrix in material frame

    BeamSection(
        double s, std::array<std::array<double, 6>, 6> mass,
        std::array<std::array<double, 6>, 6> stiffness
    )
        : position(s), M_star(mass), C_star(stiffness) {}
};

}  // namespace openturbine
