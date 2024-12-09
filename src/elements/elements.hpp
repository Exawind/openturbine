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

    /// Returns the total number of elements across all types in the system
    size_t NumElementsInSystem() const {
        return (beams ? beams->num_elems : 0) + (masses ? masses->num_elems : 0);
    }

    /**
     * @brief Returns the number of nodes per element for each element in the system
     *
     * @return Kokkos::View<size_t*> A 1D view containing the number of nodes for each element,
     *         concatenated in the order: beams / masses
     */
    Kokkos::View<size_t*> NumberOfNodesPerElement() const {
        Kokkos::View<size_t*> result("num_nodes_per_element", NumElementsInSystem());

        auto copy_with_offset =
            [&result](const Kokkos::View<size_t*>& source, const size_t offset, const size_t count) {
                auto subview =
                    Kokkos::subview(result, Kokkos::pair<size_t, size_t>(offset, offset + count));
                Kokkos::deep_copy(subview, source);
                return offset + count;
            };

        size_t current_offset{0};
        if (beams) {
            current_offset =
                copy_with_offset(beams->num_nodes_per_element, current_offset, beams->num_elems);
        }
        if (masses) {
            // Always 1 node per mass element
            Kokkos::View<size_t*> ones("ones", masses->num_elems);
            Kokkos::deep_copy(ones, 1U);
            copy_with_offset(ones, current_offset, masses->num_elems);
        }

        return result;
    }
};

}  // namespace openturbine
