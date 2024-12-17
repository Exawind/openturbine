#pragma once

#include <Kokkos_Core.hpp>

#include "src/elements/beams/beams.hpp"
#include "src/elements/masses/masses.hpp"

namespace openturbine {

/**
 * @brief A container providing handle to all structural elements present in the model
 *
 * @details This class serves as a container for wrapping all structural elements
 *          in the model, including beams and point masses, effectively providing
 *          the entire mesh as a single object to be used by the gen-alpha solver
 *          logic.
 *
 * @note The class ensures that at least one type of element (beams or masses) is present
 *       in the model
 */
struct Elements {
    std::shared_ptr<Beams> beams;
    std::shared_ptr<Masses> masses;

    Elements(std::shared_ptr<Beams> b = nullptr, std::shared_ptr<Masses> m = nullptr)
        : beams(std::move(b)), masses(std::move(m)) {
        if (beams == nullptr && masses == nullptr) {
            throw std::invalid_argument("Beams and masses cannot both be empty");
        }
    }

    /// Returns the total number of elements across all types in the system
    [[nodiscard]] size_t NumElementsInSystem() const {
        return (beams ? beams->num_elems : 0) + (masses ? masses->num_elems : 0);
    }

    /**
     * @brief Returns the number of nodes per element for each element in the system
     *
     * @return Kokkos::View<size_t*> A 1D view containing the number of nodes for each element,
     *         concatenated in the order: [beams] -> [masses] -> ...
     */
    [[nodiscard]] Kokkos::View<size_t*> NumberOfNodesPerElement() const {
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
            copy_with_offset(masses->num_nodes_per_element, current_offset, masses->num_elems);
        }
        return result;
    }

    /**
     * @brief Returns the state indices for each node of each element in the system
     *
     * @return Kokkos::View<size_t**> A 2D view containing the state indices for each node,
     *         concatenated in the order: [beams] -> [masses] -> ...
     */
    [[nodiscard]] Kokkos::View<size_t**> NodeStateIndices() const {
        const size_t max_nodes = beams ? beams->max_elem_nodes : 1;
        Kokkos::View<size_t**> result("node_state_indices", NumElementsInSystem(), max_nodes);

        auto copy_with_offset = [&result](
                                    const Kokkos::View<size_t**>& source, const size_t offset,
                                    const size_t count
                                ) {
            auto subview = Kokkos::subview(
                result, Kokkos::pair<size_t, size_t>(offset, offset + count), Kokkos::ALL()
            );
            Kokkos::deep_copy(subview, source);
            return offset + count;
        };

        size_t current_offset{0};
        if (beams) {
            current_offset =
                copy_with_offset(beams->node_state_indices, current_offset, beams->num_elems);
        }
        if (masses) {
            // Create a 2D view for masses with the same 2nd dimension as beams i.e.
            // (masses->num_elems x max_nodes) for size compatibility
            const Kokkos::View<size_t**> mass_state_indices(
                "mass_state_indices", masses->num_elems, max_nodes
            );
            // Copy the state indices from masses (1 node per element) to the temporary view
            const auto state_indices = masses->state_indices;
            Kokkos::parallel_for(
                masses->num_elems, KOKKOS_LAMBDA(const size_t i_elem
                                   ) { mass_state_indices(i_elem, 0) = state_indices(i_elem, 0); }
            );
            copy_with_offset(mass_state_indices, current_offset, masses->num_elems);
        }
        return result;
    }
};

}  // namespace openturbine
