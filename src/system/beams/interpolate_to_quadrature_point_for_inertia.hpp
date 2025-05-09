#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void InterpolateToQuadraturePointForInertia(
    const typename Kokkos::View<double*, Kokkos::LayoutLeft, DeviceType>::const_type& shape_interp,
    const typename Kokkos::View<double* [7], DeviceType>::const_type& node_u,
    const typename Kokkos::View<double* [6], DeviceType>::const_type& node_u_dot,
    const typename Kokkos::View<double* [6], DeviceType>::const_type& node_u_ddot,
    const Kokkos::View<double[4], DeviceType>& r,
    const Kokkos::View<double[3], DeviceType>& u_ddot,
    const Kokkos::View<double[3], DeviceType>& omega,
    const Kokkos::View<double[3], DeviceType>& omega_dot
) {
    for (auto i_node = 0U; i_node < node_u.extent(0); ++i_node) {
        const auto phi = shape_interp(i_node);
        for (auto k = 0U; k < 3U; ++k) {
            u_ddot(k) += node_u_ddot(i_node, k) * phi;
            omega(k) += node_u_dot(i_node, k + 3U) * phi;
            omega_dot(k) += node_u_ddot(i_node, k + 3U) * phi;
        }
        for (auto k = 0U; k < 4U; ++k) {
            r(k) += node_u(i_node, k + 3) * phi;
        }
    }
    const auto r_length = KokkosBlas::serial_nrm2(r);
    for (auto k = 0U; k < 4U; ++k) {
        r(k) /= r_length;
    }
}

}  // namespace openturbine::beams
