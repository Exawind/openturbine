#pragma once

#include <array>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct BeamSection {
    double s;          // Position of section in element on range [0, 1]
    Array_6x6 M_star;  // Mass matrix in material frame
    Array_6x6 C_star;  // Stiffness matrix in material frame

    BeamSection(
        double s, std::array<std::array<double, 6>, 6> mass,
        std::array<std::array<double, 6>, 6> stiffness
    )
        : s(s), M_star(std::move(mass)), C_star(std::move(stiffness)) {}
};

}  // namespace openturbine
