#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPPosition {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interpolation;                    // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_position_rotation;                 // Node global position vector
    View_Nx3 qp_position;                                        // quadrature point position

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto& idx = elem_indices[i_elem];
        auto shape_interp = Kokkos::subview(shape_interpolation, idx.node_range, idx.qp_shape_range);
        auto node_pos =
            Kokkos::subview(node_position_rotation, idx.node_range, Kokkos::make_pair(0, 3));
        auto qp_pos = Kokkos::subview(qp_position, idx.qp_range, Kokkos::ALL);

        for (int j = 0; j < idx.num_qps; ++j) {
            auto local_result = Kokkos::Array<double, 3>{};
            for (int i = 0; i < idx.num_nodes; ++i) {
                const auto phi = shape_interp(i, j);
                for (int k = 0; k < kVectorComponents; ++k) {
                    local_result[k] += node_pos(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_pos(j, k) = local_result[k];
            }
        }
    }
};

}  // namespace openturbine
