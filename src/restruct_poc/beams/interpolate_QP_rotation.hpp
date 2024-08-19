#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"
#include "interpolation_operations.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPRotation {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<double***>::const_type shape_interpolation;     // Num Nodes x Num Quadrature points
    Kokkos::View<double** [7]>::const_type node_position_rotation;  // Node global position vector
    Kokkos::View<double** [4]> qp_rotation;                         // quadrature point rotation

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        const auto& idx = elem_indices[i_elem];
        const auto shape_interp = Kokkos::subview(
            shape_interpolation, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_nodes),
            idx.qp_shape_range
        );
        const auto node_rot = Kokkos::subview(
            node_position_rotation, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_nodes),
            Kokkos::make_pair(3, 7)
        );
        const auto qp_rot = Kokkos::subview(
            qp_rotation, i_elem, Kokkos::make_pair(size_t{0U}, idx.num_qps), Kokkos::ALL
        );

        InterpQuaternion(shape_interp, node_rot, qp_rot);
    }
};

}  // namespace openturbine
