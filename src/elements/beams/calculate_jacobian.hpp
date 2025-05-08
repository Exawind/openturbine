#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"
#include "interpolation_operations.hpp"

namespace openturbine {

/**
 * @brief Functor to calculate Jacobians and unit tangent vectors at quadrature points for beam
 * elements
 *
 * This functor performs two main operations for a provided element:
 * - Calculates the Jacobian (J = |dx/dξ|) at each quadrature point, which represents
 *   the mapping between parametric (ξ) and physical (x) space
 * - Computes normalized position derivatives to obtain unit tangent vectors
 *
 * @note The functor modifies qp_position_derivative in place, normalizing it to create unit tangent
 * vectors
 */
template <typename DeviceType>
struct CalculateJacobian {
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t*, DeviceType>::const_type num_qps_per_element;
    typename Kokkos::View<double***, DeviceType>::const_type shape_derivative;  //< num_elems x num_nodes x num_qps
    typename Kokkos::View<double** [7], DeviceType>::const_type node_position_rotation;  //< num_elems x num_nodes x 7
    Kokkos::View<double** [3], DeviceType> qp_position_derivative;  //< output: num_elems x num_qps x 3
    Kokkos::View<double**, DeviceType> qp_jacobian;                 //< output: num_elems x num_qps

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);
        const auto shape_deriv = Kokkos::subview(
            shape_derivative, i_elem, Kokkos::make_pair(size_t{0U}, num_nodes),
            Kokkos::make_pair(size_t{0U}, num_qps)
        );
        const auto qp_pos_deriv = Kokkos::subview(
            qp_position_derivative, i_elem, Kokkos::make_pair(size_t{0U}, num_qps), Kokkos::ALL
        );
        const auto node_pos = Kokkos::subview(
            node_position_rotation, i_elem, Kokkos::make_pair(size_t{0U}, num_nodes),
            Kokkos::make_pair(0U, 3U)
        );
        const auto qp_jacob =
            Kokkos::subview(qp_jacobian, i_elem, Kokkos::make_pair(size_t{0U}, num_qps));

        // Interpolate position derivatives at quadrature points using shape functions
        // qp_pos_deriv = Σ(dN/dξ * node_pos)
        InterpVector3(shape_deriv, node_pos, qp_pos_deriv);

        for (auto j_qp = 0U; j_qp < num_qps; ++j_qp) {
            // Calculate Jacobian - this is a scalar that represents the "stretching" factor between
            // parametric (ξ) and physical (x) space
            // J = |dx/dξ| = sqrt((dx/dξ)² + (dy/dξ)² + (dz/dξ)²)
            const auto jacobian = Kokkos::sqrt(
                Kokkos::pow(qp_pos_deriv(j_qp, 0), 2.) + Kokkos::pow(qp_pos_deriv(j_qp, 1), 2.) +
                Kokkos::pow(qp_pos_deriv(j_qp, 2), 2.)
            );
            qp_jacob(j_qp) = jacobian;

            // Normalize position derivatives by Jacobian to get unit tangent vector that points
            // in the direction of curve/beam
            for (auto k_dim = 0U; k_dim < 3U; ++k_dim) {
                qp_pos_deriv(j_qp, k_dim) /= jacobian;
            }
        }
    }
};

}  // namespace openturbine
