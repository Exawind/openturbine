#pragma once

#include <Kokkos_Core.hpp>

#include "Beams.hpp"
#include "InterpolationOperations.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateJacobian {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_pos_rot_;  // Node global position/rotation vector
    View_Nx3 qp_pos_deriv_;              // quadrature point position derivative
    View_N qp_jacobian_;                 // Jacobians

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_pos_deriv = Kokkos::subview(qp_pos_deriv_, idx.qp_range, Kokkos::ALL);
        auto node_pos = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        InterpVector3(shape_deriv, node_pos, qp_pos_deriv);

        for (int j = 0; j < idx.num_qps; ++j) {
            const auto jacobian = Kokkos::sqrt(
                Kokkos::pow(qp_pos_deriv(j, 0), 2.) + Kokkos::pow(qp_pos_deriv(j, 1), 2.) +
                Kokkos::pow(qp_pos_deriv(j, 2), 2.)
            );
            qp_jacobian(j) = jacobian;
            for (int k = 0; k < 3; ++k) {
                qp_pos_deriv(j, k) /= jacobian;
            }
        }
    }
};

}  // namespace openturbine
