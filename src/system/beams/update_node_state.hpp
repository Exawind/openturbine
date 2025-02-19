#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::beams {

struct UpdateNodeStateElement {
    size_t i_elem;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<double* [7]> node_u;
    Kokkos::View<double* [6]> node_u_dot;
    Kokkos::View<double* [6]> node_u_ddot;

    Kokkos::View<double* [7]>::const_type Q;
    Kokkos::View<double* [6]>::const_type V;
    Kokkos::View<double* [6]>::const_type A;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        const auto j = node_state_indices(i_elem, i_node);
        for (auto k = 0U; k < 7U; ++k) {
            node_u(i_node, k) = Q(j, k);
        }
        for (auto k = 0U; k < 6U; ++k) {
            node_u_dot(i_node, k) = V(j, k);
            node_u_ddot(i_node, k) = A(j, k);
        }
    }
};

}  // namespace openturbine::beams
