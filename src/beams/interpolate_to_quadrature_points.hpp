#pragma once

#include <Kokkos_Core.hpp>

#include "interpolate_QP_state.hpp"
#include "interpolate_QP_vector.hpp"

namespace openturbine {

struct InterpolateToQuadraturePoints {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double***>::const_type shape_interp;
    Kokkos::View<double***>::const_type shape_deriv;
    Kokkos::View<double**>::const_type qp_jacobian;
    Kokkos::View<double** [7]>::const_type node_u;
    Kokkos::View<double** [6]>::const_type node_u_dot;
    Kokkos::View<double** [6]>::const_type node_u_ddot;
    Kokkos::View<double** [3]> qp_u;
    Kokkos::View<double** [3]> qp_uprime;
    Kokkos::View<double** [4]> qp_r;
    Kokkos::View<double** [4]> qp_rprime;
    Kokkos::View<double** [3]> qp_u_dot;
    Kokkos::View<double** [3]> qp_omega;
    Kokkos::View<double** [3]> qp_u_ddot;
    Kokkos::View<double** [3]> qp_omega_dot;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_u{i_elem, num_nodes, shape_interp, node_u, qp_u}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_uprime{i_elem, num_nodes, shape_deriv, qp_jacobian, node_u, qp_uprime}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_r{i_elem, num_nodes, shape_interp, node_u, qp_r}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_rprime{i_elem, num_nodes, shape_deriv, qp_jacobian, node_u, qp_rprime}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_dot}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(3, 6)), qp_omega}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)),
                qp_u_ddot}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(3, 6)),
                qp_omega_dot}
        );
    }
};

}  // namespace openturbine
