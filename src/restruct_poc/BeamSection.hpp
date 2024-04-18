#pragma once

#include <array>

namespace openturbine {

struct BeamSection {
    double s;                                     // Position of section in element on range [0, 1]
    std::array<std::array<double, 6>, 6> M_star;  // Mass matrix in material frame
    std::array<std::array<double, 6>, 6> C_star;  // Stiffness matrix in material frame

    BeamSection(
        double s_, std::array<std::array<double, 6>, 6> M_star_,
        std::array<std::array<double, 6>, 6> C_star_
    )
        : s(s_), M_star(std::move(M_star_)), C_star(std::move(C_star_)) {}
};

}  // namespace openturbine
