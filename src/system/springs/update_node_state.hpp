#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::springs {

/**
 * @brief Functor to update nodal states for all spring elements
 */
struct UpdateNodeState {
    Kokkos::View<size_t* [2]>::const_type node_state_indices;
    Kokkos::View<double* [3]> u1;
    Kokkos::View<double* [3]> u2;

    Kokkos::View<double* [7]>::const_type Q;

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        const auto j = node_state_indices(i_elem, 0);
        const auto k = node_state_indices(i_elem, 1);
        for (int i_dof = 0; i_dof < 3; ++i_dof) {
            u1(i_elem, i_dof) = Q(j, i_dof);  // Node 1
            u2(i_elem, i_dof) = Q(k, i_dof);  // Node 2
        }
    }
};

}  // namespace openturbine::springs
