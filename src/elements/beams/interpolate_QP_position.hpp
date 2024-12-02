#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

/**
 * @brief Interpolates quadrature point positions from nodal positions using shape functions
 *
 * This functor computes the global positions of quadrature points by interpolating
 * from the nodal positions using shape functions:
 * - Takes nodal positions and rotations for each element
 * - Uses pre-computed shape function weights for interpolation
 * - Computes the position vector (x,y,z) for each quadrature point
 */
struct InterpolateQPPosition {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double***>::const_type shape_interpolation;  //< num_elem x num_nodes x num_qps
    Kokkos::View<double** [7]>::const_type
        node_position_rotation;  //< node position vector/generalized coordinates in global csys
    Kokkos::View<double** [3]> qp_position;  //< quadrature point position - x, y, z (computed)

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);
        for (auto j = 0U; j < num_qps; ++j) {
            auto local_result = Kokkos::Array<double, 3>{};
            for (auto i = 0U; i < num_nodes; ++i) {
                const auto phi = shape_interpolation(i_elem, i, j);
                for (auto k = 0U; k < kVectorComponents; ++k) {
                    local_result[k] += node_position_rotation(i_elem, i, k) * phi;
                }
            }
            for (auto k = 0U; k < 3U; ++k) {
                qp_position(i_elem, j, k) = local_result[k];
            }
        }
    }
};

}  // namespace openturbine
