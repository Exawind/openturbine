#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::masses {

/**
 * @brief Functor to update nodal states for all elements in the mesh
 *
 * Iterates over all elements, updating the nodal states for each element using the provided
 * global state vectors.
 */
struct UpdateNodeState {
    Kokkos::View<size_t*>::const_type node_state_indices;
    Kokkos::View<double* [7]> node_u;
    Kokkos::View<double* [6]> node_u_dot;
    Kokkos::View<double* [6]> node_u_ddot;

    Kokkos::View<double* [7]>::const_type Q;
    Kokkos::View<double* [6]>::const_type V;
    Kokkos::View<double* [6]>::const_type A;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        const auto j = node_state_indices(i_elem);
        for (auto k = 0U; k < 7U; ++k) {
            node_u(i_elem, k) = Q(j, k);
        }
        for (auto k = 0U; k < 6U; ++k) {
            node_u_dot(i_elem, k) = V(j, k);
            node_u_ddot(i_elem, k) = A(j, k);
        }
    }
};

}  // namespace openturbine
