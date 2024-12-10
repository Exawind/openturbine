#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

/**
 * @brief Functor to update nodal states for a single beamelement from global state vectors
 */
struct UpdateNodeStateBeamElement {
    size_t i_elem;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<double** [7]> node_u;
    Kokkos::View<double** [6]> node_u_dot;
    Kokkos::View<double** [6]> node_u_ddot;

    Kokkos::View<double* [7]>::const_type Q;
    Kokkos::View<double* [6]>::const_type V;
    Kokkos::View<double* [6]>::const_type A;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        const auto j = node_state_indices(i_elem, i_node);
        for (auto k = 0U; k < 7U; ++k) {
            node_u(i_elem, i_node, k) = Q(j, k);
        }
        for (auto k = 0U; k < 6U; ++k) {
            node_u_dot(i_elem, i_node, k) = V(j, k);
            node_u_ddot(i_elem, i_node, k) = A(j, k);
        }
    }
};

/**
 * @brief Functor to update nodal states for all elements in the mesh
 *
 * Iterates over all elements, updating the nodal states for each element using the provided
 * global state vectors.
 */
struct UpdateNodeState {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<double** [7]> node_u;
    Kokkos::View<double** [6]> node_u_dot;
    Kokkos::View<double** [6]> node_u_ddot;

    Kokkos::View<double* [7]>::const_type Q;
    Kokkos::View<double* [6]>::const_type V;
    Kokkos::View<double* [6]>::const_type A;

    KOKKOS_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type& member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_nodes),
            UpdateNodeStateBeamElement{
                i_elem, node_state_indices, node_u, node_u_dot, node_u_ddot, Q, V, A
            }
        );
    }
};

/**
 * @brief Functor to update node states for a single mass element from global state vectors
 */
struct UpdateNodeStateMassElement {
    size_t i_elem;
    Kokkos::View<size_t*>::const_type state_indices;
    Kokkos::View<double* [7]> u;
    Kokkos::View<double* [6]> u_dot;
    Kokkos::View<double* [6]> u_ddot;

    Kokkos::View<double* [7]>::const_type Q;
    Kokkos::View<double* [6]>::const_type V;
    Kokkos::View<double* [6]>::const_type A;

    KOKKOS_FUNCTION
    void operator()(const size_t) const {
        const auto j = state_indices(i_elem);
        for (auto k = 0U; k < 7U; ++k) {
            u(i_elem, k) = Q(j, k);
        }
        for (auto k = 0U; k < 6U; ++k) {
            u_dot(i_elem, k) = V(j, k);
            u_ddot(i_elem, k) = A(j, k);
        }
    }
};

}  // namespace openturbine
