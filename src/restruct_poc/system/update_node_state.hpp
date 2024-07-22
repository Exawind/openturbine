#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct UpdateNodeState {
    Kokkos::View<int*>::const_type node_state_indices;
    View_Nx7 node_u;
    View_Nx6 node_u_dot;
    View_Nx6 node_u_ddot;

    View_Nx7::const_type Q;
    View_Nx6::const_type V;
    View_Nx6::const_type A;

    KOKKOS_FUNCTION
    void operator()(int i) const {
        auto j = node_state_indices(i);
        for (int k = 0; k < kLieGroupComponents; k++) {
            node_u(i, k) = Q(j, k);
        }
        for (int k = 0; k < kLieAlgebraComponents; k++) {
            node_u_dot(i, k) = V(j, k);
        }
        for (int k = 0; k < kLieAlgebraComponents; k++) {
            node_u_ddot(i, k) = A(j, k);
        }
    }
};

}  // namespace openturbine
