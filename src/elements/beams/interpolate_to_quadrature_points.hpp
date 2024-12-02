#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_QP_position.hpp"
#include "interpolate_QP_state.hpp"
#include "interpolate_QP_vector.hpp"

namespace openturbine {

/**
 * @brief Interpolates various quantities from nodes to quadrature points for beam elements
 *
 * This functor handles the interpolation of multiple quantities from nodal points to
 * quadrature points (QPs) for beam elements, including:
 * - Displacements (u) and their derivatives
 * - Rotations (r) and their derivatives
 * - Velocities (u_dot) and angular velocities (omega)
 * - Accelerations (u_ddot) and angular accelerations (omega_dot)
 * - Final positions (x) of quadrature points
 */
struct InterpolateToQuadraturePoints {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;  //< Number of nodes per element
    Kokkos::View<size_t*>::const_type num_qps_per_element;    //< Number of QPs per element
    Kokkos::View<double***>::const_type shape_interp;         //< shape functions at QPs
    Kokkos::View<double***>::const_type shape_deriv;          //< shape function derivatives at QPs
    Kokkos::View<double**>::const_type qp_jacobian;           //< Jacobian at QPs
    Kokkos::View<double** [7]>::const_type node_u;            //< Nodal displacements
    Kokkos::View<double** [6]>::const_type node_u_dot;        //< Nodal velocities
    Kokkos::View<double** [6]>::const_type node_u_ddot;       //< Nodal accelerations
    Kokkos::View<double** [3]>::const_type qp_x0;             //< Initial positions at QPs
    Kokkos::View<double** [4]>::const_type qp_r0;             //< Initial rotations at QPs
    // Output quantities at quadrature points
    Kokkos::View<double** [3]> qp_u;          //< Interpolated displacements at QPs
    Kokkos::View<double** [3]> qp_uprime;     //< Displacement derivatives at QPs
    Kokkos::View<double** [4]> qp_r;          //< Interpolated rotations at QPs
    Kokkos::View<double** [4]> qp_rprime;     //< Rotation derivatives at QPs
    Kokkos::View<double** [3]> qp_u_dot;      //< Interpolated velocities at QPs
    Kokkos::View<double** [3]> qp_omega;      //< Interpolated angular velocities at QPs
    Kokkos::View<double** [3]> qp_u_ddot;     //< Interpolated accelerations at QPs
    Kokkos::View<double** [3]> qp_omega_dot;  //< Interpolated angular accelerations at QPs
    Kokkos::View<double** [7]> qp_x;          //< Final positions of quadrature points

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
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_dot
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(3, 6)), qp_omega
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_ddot
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(3, 6)),
                qp_omega_dot
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateQPPosition{i_elem, qp_x0, qp_u, qp_r0, qp_r, qp_x}
        );
    }
};

}  // namespace openturbine
