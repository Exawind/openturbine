#pragma once

#include "mass_element.hpp"

namespace openturbine {

/**
 * @brief Represents the input data for creating mass/rigid body elements
 */
struct MassesInput {
    std::vector<MassElement> elements;  //< All elements present in the masses/rigid_body
    std::array<double, 3> gravity;      //< Gravity vector

    MassesInput(std::vector<MassElement> elems, std::array<double, 3> g)
        : elements(std::move(elems)), gravity(g) {}

    /// Returns the total number of elements present in masses/rigid_body portion of the mesh
    [[nodiscard]] size_t NumElements() const { return elements.size(); }
};

}  // namespace openturbine
