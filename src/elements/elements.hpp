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
    Beams beams;
    Masses masses;

    Elements(Beams b, Masses m) : beams(std::move(b)), masses(std::move(m)) {}

    /// Returns the total number of elements across all types in the system
    [[nodiscard]] size_t NumElementsInSystem() const { return beams.num_elems + masses.num_elems; }

    /**
     * @brief Returns the number of nodes per element for each element in the system
     *
     * @return Kokkos::View<size_t*> A 1D view containing the number of nodes for each element,
     *         concatenated in the order: [beams] -> [masses] -> ...
     */
    [[nodiscard]] Kokkos::View<size_t*> NumberOfNodesPerElement() const {
        Kokkos::View<size_t*> result("num_nodes_per_element", NumElementsInSystem());

        auto beams_num_nodes_per_element = beams.num_nodes_per_element;
        Kokkos::parallel_for(
            beams.num_elems,
            KOKKOS_LAMBDA(size_t i_elem) { result(i_elem) = beams_num_nodes_per_element(i_elem); }
        );
        auto beams_offset = beams.num_elems;
        auto masses_num_nodes_per_element = masses.num_nodes_per_element;
        Kokkos::parallel_for(
            masses.num_elems,
            KOKKOS_LAMBDA(size_t i_elem) {
                result(i_elem + beams_offset) = masses_num_nodes_per_element(i_elem);
            }
        );

        return result;
    }

    /**
     * @brief Returns the state indices for each node of each element in the system
     *
     * @return Kokkos::View<size_t**> A 2D view containing the state indices for each node,
     *         concatenated in the order: [beams] -> [masses] -> ...
     */
    [[nodiscard]] Kokkos::View<size_t**> NodeStateIndices() const {
        const auto max_nodes = std::max(beams.max_elem_nodes, 1UL);
        Kokkos::View<size_t**> result("node_state_indices", NumElementsInSystem(), max_nodes);

        auto beams_num_nodes_per_element = beams.num_nodes_per_element;
        auto beams_node_state_indices = beams.node_state_indices;
        Kokkos::parallel_for(
            beams.num_elems,
            KOKKOS_LAMBDA(size_t i_elem) {
                const auto num_nodes = beams_num_nodes_per_element(i_elem);
                for (auto j = 0U; j < num_nodes; ++j) {
                    result(i_elem, j) = beams_node_state_indices(i_elem, j);
                }
            }
        );
        const auto beams_offset = beams.num_elems;
        auto masses_state_indices = masses.state_indices;
        Kokkos::parallel_for(
            masses.num_elems, KOKKOS_LAMBDA(size_t i_elem
                              ) { result(i_elem + beams_offset, 0) = masses_state_indices(i_elem); }
        );

        return result;
    }
};

}  // namespace openturbine
