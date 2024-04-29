#pragma once

#include <Kokkos_Core.hpp>

#include "InterpolationOperations.hpp"
#include "Beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPRotation {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_pos_rot_;                          // Node global position vector
    View_Nx4 qp_rot_;                                            // quadrature point rotation

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interp_, idx.node_range, idx.qp_shape_range);
        auto node_rot = Kokkos::subview(node_pos_rot_, idx.node_range, Kokkos::make_pair(3, 7));
        auto qp_rot = Kokkos::subview(qp_rot_, idx.qp_range, Kokkos::ALL);

        InterpQuaternion(shape_interp, node_rot, qp_rot);
    }
};

}  // namespace openturbine
