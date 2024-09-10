#pragma once

#include <Kokkos_Core.hpp>

#include "beams_data.hpp"
#include "interpolation_operations.hpp"
#include "types.hpp"

namespace openturbine {

struct InterpolateQPRotationDerivative {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<double***>::const_type shape_derivative;        // Num Nodes x Num Quadrature points
    View_N::const_type qp_jacobian;                              // Jacobians
    Kokkos::View<double** [7]>::const_type node_pos_rot;  // Node global position/rotation vector
    Kokkos::View<double** [4]> qp_rot_deriv;              // quadrature point rotation derivative

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_deriv = Kokkos::subview(
            shape_derivative, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_nodes),
            idx.qp_shape_range
        );
        auto qp_rot_deriv = Kokkos::subview(qp_rot_deriv, idx.qp_range, Kokkos::ALL);
        auto node_rot = Kokkos::subview(
            node_pos_rot, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_nodes),
            Kokkos::make_pair(3, 7)
        );
        auto qp_jac = Kokkos::subview(qp_jacobian, idx.qp_range);

        InterpVector4Deriv(shape_deriv, qp_jac, node_rot, qp_rot_deriv);
    }
};

}  // namespace openturbine
