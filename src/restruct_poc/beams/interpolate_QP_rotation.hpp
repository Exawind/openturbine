#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"
#include "interpolation_operations.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPRotation {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interpolation;                    // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_position_rotation;                 // Node global position vector
    View_Nx4 qp_rotation;                                        // quadrature point rotation

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interpolation, idx.node_range, idx.qp_shape_range);
        auto node_rot =
            Kokkos::subview(node_position_rotation, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_rot = Kokkos::subview(qp_rotation, idx.qp_range, Kokkos::ALL);

        InterpQuaternion(shape_interp, node_rot, qp_rot);
    }
};

}  // namespace openturbine
