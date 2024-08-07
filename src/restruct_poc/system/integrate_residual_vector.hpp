#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateResidualVector {
    Kokkos::View<size_t*>::const_type node_state_indices_;
    View_Nx6::const_type node_FE_;  // Elastic force
    View_Nx6::const_type node_FI_;  // Inertial force
    View_Nx6::const_type node_FG_;  // Gravity force
    View_Nx6::const_type node_FX_;  // External force
    View_N_atomic residual_vector_;

    KOKKOS_INLINE_FUNCTION void operator()(const int i_node) const {
        const auto i_rv_start = node_state_indices_(i_node) * kLieAlgebraComponents;
        for (auto j = 0U; j < 6U; j++) {
            residual_vector_(i_rv_start + j) += node_FE_(i_node, j) + node_FI_(i_node, j) -
                                                node_FX_(i_node, j) - node_FG_(i_node, j);
        }
    }
};

}  // namespace openturbine
