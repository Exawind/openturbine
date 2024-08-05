#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"
#include "interpolation_operations.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateJacobian {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<double***>::const_type shape_derivative;        // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_position_rotation;  // Node global position/rotation vector
    View_Nx3 qp_position_derivative;              // quadrature point position derivative
    View_NxN qp_jacobian;                         // Jacobians

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        const auto& idx = elem_indices[i_elem];
        const auto shape_deriv = Kokkos::subview(
            shape_derivative, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_nodes),
            idx.qp_shape_range
        );
        const auto qp_pos_deriv = Kokkos::subview(qp_position_derivative, idx.qp_range, Kokkos::ALL);
        const auto node_pos =
            Kokkos::subview(node_position_rotation, idx.node_range, Kokkos::make_pair(0, 3));
        const auto qp_jacob =
            Kokkos::subview(qp_jacobian, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_qps));

        InterpVector3(shape_deriv, node_pos, qp_pos_deriv);

        for (auto j = 0U; j < idx.num_qps; ++j) {
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
