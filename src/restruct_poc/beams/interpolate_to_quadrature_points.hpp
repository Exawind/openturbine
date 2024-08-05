#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"
#include "interpolate_QP_state.hpp"
#include "interpolate_QP_vector.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateToQuadraturePoints {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    Kokkos::View<double***>::const_type shape_interp;
    Kokkos::View<double***>::const_type shape_deriv;
    View_NxN::const_type qp_jacobian;
    View_Nx7::const_type node_u;
    View_Nx6::const_type node_u_dot;
    View_Nx6::const_type node_u_ddot;
    View_Nx3 qp_u;
    View_Nx3 qp_uprime;
    View_Nx4 qp_r;
    View_Nx4 qp_rprime;
    View_Nx3 qp_u_dot;
    View_Nx3 qp_omega;
    View_Nx3 qp_u_ddot;
    View_Nx3 qp_omega_dot;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto idx = elem_indices(i_elem);
        const auto first_qp = idx.qp_range.first;
        const auto first_node = idx.node_range.first;
        const auto num_nodes = idx.num_nodes;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPState_u{i_elem, first_qp, first_node, num_nodes, shape_interp, node_u, qp_u}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPState_uprime{
                i_elem, first_qp, first_node, num_nodes, shape_deriv, qp_jacobian, node_u, qp_uprime}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPState_r{i_elem, first_qp, first_node, num_nodes, shape_interp, node_u, qp_r}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPState_rprime{
                i_elem, first_qp, first_node, num_nodes, shape_deriv, qp_jacobian, node_u, qp_rprime}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPVector{
                i_elem, first_qp, first_node, num_nodes, shape_interp,
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_dot}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPVector{
                i_elem, first_qp, first_node, num_nodes, shape_interp,
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::pair(3, 6)), qp_omega}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPVector{
                i_elem, first_qp, first_node, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_ddot}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_qps),
            InterpolateQPVector{
                i_elem, first_qp, first_node, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::pair(3, 6)), qp_omega_dot}
        );
    }
};

}  // namespace openturbine