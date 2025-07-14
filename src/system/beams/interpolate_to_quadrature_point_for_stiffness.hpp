#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void InterpolateToQuadraturePointForStiffness(
    double jacobian,
    const typename Kokkos::View<double*, Kokkos::LayoutLeft, DeviceType>::const_type& shape_interp,
    const typename Kokkos::View<double*, Kokkos::LayoutLeft, DeviceType>::const_type& shape_deriv,
    const typename Kokkos::View<double* [7], DeviceType>::const_type& node_u,
    const Kokkos::View<double[3], DeviceType>& u, const Kokkos::View<double[4], DeviceType>& r,
    const Kokkos::View<double[3], DeviceType>& u_prime,
    const Kokkos::View<double[4], DeviceType>& r_prime
) {
    for (auto node = 0U; node < node_u.extent(0); ++node) {
        const auto phi = shape_interp(node);
        const auto dphiJ = shape_deriv(node) / jacobian;
        for (auto component = 0U; component < 3U; ++component) {
            u(component) += node_u(node, component) * phi;
            u_prime(component) += node_u(node, component) * dphiJ;
        }
        for (auto component = 0U; component < 4U; ++component) {
            r(component) += node_u(node, component + 3) * phi;
            r_prime(component) += node_u(node, component + 3) * dphiJ;
        }
    }
    const auto r_length = KokkosBlas::serial_nrm2(r);
    for (auto component = 0U; component < 4U; ++component) {
        r(component) /= r_length;
    }
}

}  // namespace openturbine::beams
