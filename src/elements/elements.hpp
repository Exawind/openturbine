#pragma once

#include <Kokkos_Core.hpp>

#include "src/elements/beams/beams.hpp"
#include "src/elements/masses/masses.hpp"

namespace openturbine {

/**
 * @brief A container for all structural elements present in a model
 */
struct Elements {
    std::shared_ptr<Beams> beams;
    std::shared_ptr<Masses> masses;

    Elements(std::shared_ptr<Beams> beams = nullptr, std::shared_ptr<Masses> masses = nullptr)
        : beams(beams), masses(masses) {
        if (beams == nullptr && masses == nullptr) {
            throw std::invalid_argument("Beams and masses cannot both be empty");
        }
    }
};

}  // namespace openturbine
