#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "mass_element.hpp"

namespace openturbine {

struct MassesInput {
    std::vector<MassElement> elements;
    std::array<double, 3> gravity;

    MassesInput(std::vector<MassElement> elems, std::array<double, 3> g)
        : elements(std::move(elems)), gravity(std::move(g)) {}

    size_t NumNodes() const { return elements.size(); };
};

}  // namespace openturbine
