#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateResidualVectorElement {
    size_t i_elem;
    Kokkos::View<size_t**>::const_type node_state_indices_;
    Kokkos::View<double** [6]>::const_type node_FE_;  // Elastic force
    Kokkos::View<double** [6]>::const_type node_FI_;  // Inertial force
    Kokkos::View<double** [6]>::const_type node_FG_;  // Gravity force
    Kokkos::View<double** [6]>::const_type node_FX_;  // External force
    View_N_atomic residual_vector_;

    KOKKOS_FUNCTION void operator()(size_t i_node) const {
        const auto node_start = node_state_indices_(i_elem, i_node) * kLieAlgebraComponents;
        for (auto j = 0U; j < 6U; ++j) {
            residual_vector_(node_start + j) +=
                node_FE_(i_elem, i_node, j) + node_FI_(i_elem, i_node, j) -
                node_FX_(i_elem, i_node, j) - node_FG_(i_elem, i_node, j);
        }
    }
};

struct IntegrateResidualVector {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices_;
    Kokkos::View<double** [6]>::const_type node_FE_;  // Elastic force
    Kokkos::View<double** [6]>::const_type node_FI_;  // Inertial force
    Kokkos::View<double** [6]>::const_type node_FG_;  // Gravity force
    Kokkos::View<double** [6]>::const_type node_FX_;  // External force
    View_N_atomic residual_vector_;

    KOKKOS_FUNCTION void operator()(const Kokkos::TeamPolicy<>::member_type& member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_nodes),
            IntegrateResidualVectorElement{
                i_elem, node_state_indices_, node_FE_, node_FI_, node_FG_, node_FX_,
                residual_vector_}
        );
    }
};

}  // namespace openturbine
