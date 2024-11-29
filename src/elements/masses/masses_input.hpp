#pragma once

#include <array>
#include <vector>

#include "mass_element.hpp"

namespace openturbine {

struct MassesInput {
    std::vector<MassElement> elements;  //< Elements in the masses/rigid_body
    std::array<double, 3> gravity;      //< Gravity vector

    MassesInput(std::vector<MassElement> elems, std::array<double, 3> g)
        : elements(std::move(elems)), gravity(g) {}

    /// Returns the number of elements in the beam
    [[nodiscard]] size_t NumElements() const { return elements.size(); }
};

}  // namespace openturbine
