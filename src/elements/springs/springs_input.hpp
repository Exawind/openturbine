#pragma once

#include "spring_element.hpp"

namespace openturbine {

/**
 * @brief Represents the input data for creating spring elements
 */
struct SpringsInput {
    std::vector<SpringElement> elements;  //< All spring elements in the system

    explicit SpringsInput(std::vector<SpringElement> elems) : elements(std::move(elems)) {}

    /// Returns the total number of spring elements in the system
    [[nodiscard]] size_t NumElements() const { return elements.size(); }
};

}  // namespace openturbine
