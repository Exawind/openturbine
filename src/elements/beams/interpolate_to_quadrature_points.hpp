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
template <typename DeviceType>
struct InterpolateToQuadraturePoints {
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;  //< Number of nodes per element
    typename Kokkos::View<size_t*, DeviceType>::const_type num_qps_per_element;    //< Number of QPs per element
    typename Kokkos::View<double***, DeviceType>::const_type shape_interp;         //< shape functions at QPs
    typename Kokkos::View<double***, DeviceType>::const_type shape_deriv;          //< shape function derivatives at QPs
    typename Kokkos::View<double**, DeviceType>::const_type qp_jacobian;           //< Jacobian at QPs
    typename Kokkos::View<double** [7], DeviceType>::const_type node_u;            //< Nodal displacements
    typename Kokkos::View<double** [6], DeviceType>::const_type node_u_dot;        //< Nodal velocities
    typename Kokkos::View<double** [6], DeviceType>::const_type node_u_ddot;       //< Nodal accelerations
    typename Kokkos::View<double** [3], DeviceType>::const_type qp_x0;             //< Initial positions at QPs
    typename Kokkos::View<double** [4], DeviceType>::const_type qp_r0;             //< Initial rotations at QPs
    // Output quantities at quadrature points
    Kokkos::View<double** [3], DeviceType> qp_u;          //< Interpolated displacements at QPs
    Kokkos::View<double** [3], DeviceType> qp_uprime;     //< Displacement derivatives at QPs
    Kokkos::View<double** [4], DeviceType> qp_r;          //< Interpolated rotations at QPs
    Kokkos::View<double** [4], DeviceType> qp_rprime;     //< Rotation derivatives at QPs
    Kokkos::View<double** [3], DeviceType> qp_u_dot;      //< Interpolated velocities at QPs
    Kokkos::View<double** [3], DeviceType> qp_omega;      //< Interpolated angular velocities at QPs
    Kokkos::View<double** [3], DeviceType> qp_u_ddot;     //< Interpolated accelerations at QPs
    Kokkos::View<double** [3], DeviceType> qp_omega_dot;  //< Interpolated angular accelerations at QPs
    Kokkos::View<double** [7], DeviceType> qp_x;          //< Final positions of quadrature points

    KOKKOS_FUNCTION
    void operator()(typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_u<DeviceType>{i_elem, num_nodes, shape_interp, node_u, qp_u}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_uprime<DeviceType>{i_elem, num_nodes, shape_deriv, qp_jacobian, node_u, qp_uprime}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_r<DeviceType>{i_elem, num_nodes, shape_interp, node_u, qp_r}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPState_rprime<DeviceType>{i_elem, num_nodes, shape_deriv, qp_jacobian, node_u, qp_rprime}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector<DeviceType>{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_dot
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector<DeviceType>{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(3, 6)), qp_omega
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector<DeviceType>{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_ddot
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            InterpolateQPVector<DeviceType>{
                i_elem, num_nodes, shape_interp,
                Kokkos::subview(node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(3, 6)),
                qp_omega_dot
            }
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateQPPosition<DeviceType>{i_elem, qp_x0, qp_u, qp_r0, qp_r, qp_x}
        );
    }
};

}  // namespace openturbine
