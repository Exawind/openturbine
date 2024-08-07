#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPPosition {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<double***>::const_type shape_interpolation;     // Num Nodes x Num Quadrature points
    View_Nx7::const_type node_position_rotation;                 // Node global position vector
    Kokkos::View<double** [3]> qp_position;                      // quadrature point position

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        const auto& idx = elem_indices[i_elem];
        const auto shape_interp = Kokkos::subview(
            shape_interpolation, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_nodes),
            idx.qp_shape_range
        );
        const auto node_pos =
            Kokkos::subview(node_position_rotation, idx.node_range, Kokkos::make_pair(0, 3));
        const auto qp_pos = Kokkos::subview(
            qp_position, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_qps), Kokkos::ALL
        );

        for (auto j = 0U; j < idx.num_qps; ++j) {
            auto local_result = Kokkos::Array<double, 3>{};
            for (auto i = 0U; i < idx.num_nodes; ++i) {
                const auto phi = shape_interp(i, j);
                for (auto k = 0U; k < kVectorComponents; ++k) {
                    local_result[k] += node_pos(i, k) * phi;
                }
            }
            for (auto k = 0U; k < 3U; ++k) {
                qp_pos(j, k) = local_result[k];
            }
        }
    }
};

}  // namespace openturbine
