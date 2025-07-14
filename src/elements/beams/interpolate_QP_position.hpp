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
template <typename DeviceType>
struct InterpolateQPPosition {
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t*, DeviceType>::const_type num_qps_per_element;
    typename Kokkos::View<double***, DeviceType>::const_type
        shape_interpolation;  //< num_elem x num_nodes x num_qps
    typename Kokkos::View<double** [7], DeviceType>::const_type
        node_position_rotation;  //< node position vector/generalized coordinates in global csys
    Kokkos::View<double** [3], DeviceType>
        qp_position;  //< quadrature point position - x, y, z (computed)

    KOKKOS_FUNCTION
    void operator()(const int element) const {
        const auto num_nodes = num_nodes_per_element(element);
        const auto num_qps = num_qps_per_element(element);
        for (auto qp = 0U; qp < num_qps; ++qp) {
            auto local_result = Kokkos::Array<double, 3>{};
            for (auto node = 0U; node < num_nodes; ++node) {
                const auto phi = shape_interpolation(element, node, qp);
                for (auto component = 0U; component < 3U; ++component) {
                    local_result[component] +=
                        node_position_rotation(element, node, component) * phi;
                }
            }
            for (auto component = 0U; component < 3U; ++component) {
                qp_position(element, qp, component) = local_result[component];
            }
        }
    }
};

}  // namespace openturbine
