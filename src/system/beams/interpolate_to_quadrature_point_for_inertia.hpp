#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::beams {

KOKKOS_FUNCTION
inline void InterpolateToQuadraturePointForInertia(
    const Kokkos::View<double*, Kokkos::LayoutLeft>::const_type& shape_interp,
    const Kokkos::View<double* [7]>::const_type& node_u,
    const Kokkos::View<double* [6]>::const_type& node_u_dot,
    const Kokkos::View<double* [6]>::const_type& node_u_ddot, const Kokkos::View<double[4]>& r,
    const Kokkos::View<double[3]>& u_ddot, const Kokkos::View<double[3]>& omega,
    const Kokkos::View<double[3]>& omega_dot
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
