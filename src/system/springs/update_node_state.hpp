#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::springs {

/**
 * @brief Functor to update the nodal displacement vectors for all spring elements based on the
 * current state Q
 */
struct UpdateNodeState {
    Kokkos::View<size_t* [2]>::const_type node_state_indices;  //< Node indices
    Kokkos::View<double* [3]> u1;                              //< Displacement of node 1
    Kokkos::View<double* [3]> u2;                              //< Displacement of node 2
    Kokkos::View<double* [7]>::const_type Q;                   //< State vector

    KOKKOS_FUNCTION
    void operator()(const size_t i_elem) const {
        const auto node1_index = node_state_indices(i_elem, 0);
        const auto node2_index = node_state_indices(i_elem, 1);
        for (int i_dof = 0; i_dof < 3; ++i_dof) {
            u1(i_elem, i_dof) = Q(node1_index, i_dof);
            u2(i_elem, i_dof) = Q(node2_index, i_dof);
        }
    }
};

}  // namespace openturbine::springs
