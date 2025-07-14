#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
struct UpdateNodeStateElement {
    size_t element{};
    typename Kokkos::View<size_t**, DeviceType>::const_type node_state_indices;
    Kokkos::View<double* [7], DeviceType> node_u;
    Kokkos::View<double* [6], DeviceType> node_u_dot;
    Kokkos::View<double* [6], DeviceType> node_u_ddot;

    typename Kokkos::View<double* [7], DeviceType>::const_type Q;
    typename Kokkos::View<double* [6], DeviceType>::const_type V;
    typename Kokkos::View<double* [6], DeviceType>::const_type A;

    KOKKOS_FUNCTION
    void operator()(const size_t node) const {
        const auto index = node_state_indices(element, node);
        for (auto component = 0U; component < 7U; ++component) {
            node_u(node, component) = Q(index, component);
        }
        for (auto component = 0U; component < 6U; ++component) {
            node_u_dot(node, component) = V(index, component);
            node_u_ddot(node, component) = A(index, component);
        }
    }
};

template <typename DeviceType>
struct UpdateNodeState {
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    typename Kokkos::View<double* [7]>::const_type Q;
    typename Kokkos::View<double* [6]>::const_type V;
    typename Kokkos::View<double* [6]>::const_type A;
    typename Kokkos::View<size_t**>::const_type node_state_indices;
    typename Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<double** [7]> node_u_;
    Kokkos::View<double** [6]> node_u_dot_;
    Kokkos::View<double** [6]> node_u_ddot_;

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        const auto element = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(element);
        const auto node_range = Kokkos::TeamThreadRange(member, num_nodes);

        const auto node_u = Kokkos::View<double* [7], DeviceType>(member.team_scratch(1), num_nodes);
        const auto node_u_dot =
            Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_nodes);
        const auto node_u_ddot =
            Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_nodes);

        const auto node_state_updater = beams::UpdateNodeStateElement<DeviceType>{
            element, node_state_indices, node_u, node_u_dot, node_u_ddot, Q, V, A
        };
        Kokkos::parallel_for(node_range, node_state_updater);

        KokkosBatched::TeamVectorCopy<Kokkos::TeamPolicy<>::member_type>::invoke(
            member, node_u, Kokkos::subview(node_u_, element, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::TeamVectorCopy<Kokkos::TeamPolicy<>::member_type>::invoke(
            member, node_u_dot, Kokkos::subview(node_u_dot_, element, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::TeamVectorCopy<Kokkos::TeamPolicy<>::member_type>::invoke(
            member, node_u_ddot, Kokkos::subview(node_u_ddot_, element, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine::beams
