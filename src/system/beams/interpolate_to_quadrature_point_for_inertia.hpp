#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void InterpolateToQuadraturePointForInertia(
    const typename Kokkos::View<double*, Kokkos::LayoutLeft, DeviceType>::const_type& shape_interp,
    const typename Kokkos::View<double* [7], DeviceType>::const_type& node_u,
    const typename Kokkos::View<double* [6], DeviceType>::const_type& node_u_dot,
    const typename Kokkos::View<double* [6], DeviceType>::const_type& node_u_ddot,
    const Kokkos::View<double[4], DeviceType>& r, const Kokkos::View<double[3], DeviceType>& u_ddot,
    const Kokkos::View<double[3], DeviceType>& omega,
    const Kokkos::View<double[3], DeviceType>& omega_dot
) {
    for (auto node = 0U; node < node_u.extent(0); ++node) {
        const auto phi = shape_interp(node);
        for (auto component = 0U; component < 3U; ++component) {
            u_ddot(component) += node_u_ddot(node, component) * phi;
            omega(component) += node_u_dot(node, component + 3U) * phi;
            omega_dot(component) += node_u_ddot(node, component + 3U) * phi;
        }
        for (auto component = 0U; component < 4U; ++component) {
            r(component) += node_u(node, component + 3) * phi;
        }
    }
    const auto r_length = KokkosBlas::serial_nrm2(r);
    for (auto component = 0U; component < 4U; ++component) {
        r(component) /= r_length;
    }
}

}  // namespace openturbine::beams
