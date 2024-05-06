#pragma once

#include <Kokkos_Core.hpp>

#include "beams_data.hpp"
#include "interpolation_operations.hpp"
#include "types.hpp"

namespace openturbine {

struct InterpolateQPRotationDerivative {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_Nx7::const_type node_pos_rot_;  // Node global position/rotation vector
    View_Nx4 qp_rot_deriv_;              // quadrature point rotation derivative

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_deriv = Kokkos::subview(shape_deriv_, idx.node_range, idx.qp_shape_range);
        auto qp_rot_deriv = Kokkos::subview(qp_rot_deriv_, idx.qp_range, Kokkos::ALL);
        auto node_rot = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_jacobian = Kokkos::subview(qp_jacobian_, idx.qp_range);

        InterpVector4Deriv(shape_deriv, qp_jacobian, node_rot, qp_rot_deriv);
    }
};

}  // namespace openturbine
