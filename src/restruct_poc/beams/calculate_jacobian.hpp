#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"
#include "interpolation_operations.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateJacobian {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double***>::const_type shape_derivative;  // Num Nodes x Num Quadrature points
    Kokkos::View<double** [7]>::const_type
        node_position_rotation;                         // Node global position/rotation vector
    Kokkos::View<double** [3]> qp_position_derivative;  // quadrature point position derivative
    Kokkos::View<double**> qp_jacobian;                 // Jacobians

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

        InterpVector3(shape_deriv, node_pos, qp_pos_deriv);

        for (auto j = 0U; j < num_qps; ++j) {
            const auto jacobian = Kokkos::sqrt(
                Kokkos::pow(qp_pos_deriv(j, 0), 2.) + Kokkos::pow(qp_pos_deriv(j, 1), 2.) +
                Kokkos::pow(qp_pos_deriv(j, 2), 2.)
            );
            qp_jacob(j) = jacobian;
            for (auto k = 0U; k < 3U; ++k) {
                qp_pos_deriv(j, k) /= jacobian;
            }
        }
    }
};

}  // namespace openturbine
