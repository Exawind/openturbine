#pragma once

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
    for (auto i_node = 0U; i_node < node_u.extent(0); ++i_node) {
        const auto phi = shape_interp(i_node);
        const auto dphiJ = shape_deriv(i_node) / jacobian;
        for (auto k = 0U; k < 3U; ++k) {
            u(k) += node_u(i_node, k) * phi;
            u_prime(k) += node_u(i_node, k) * dphiJ;
        }
        for (auto k = 0U; k < 4U; ++k) {
            r(k) += node_u(i_node, k + 3) * phi;
            r_prime(k) += node_u(i_node, k + 3) * dphiJ;
        }
    }
    const auto r_length = KokkosBlas::serial_nrm2(r);
    for (auto k = 0U; k < 4U; ++k) {
        r(k) /= r_length;
    }
}

}  // namespace openturbine::beams
