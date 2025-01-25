#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::masses {

struct CopyToQuadraturePoints {
    Kokkos::View<double* [7]>::const_type node_x0;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]>::const_type node_u_dot;
    Kokkos::View<double* [6]>::const_type node_u_ddot;

    Kokkos::View<double* [3]> qp_x0;
    Kokkos::View<double* [4]> qp_r0;
    Kokkos::View<double* [3]> qp_u;
    Kokkos::View<double* [4]> qp_r;
    Kokkos::View<double* [3]> qp_u_ddot;
    Kokkos::View<double* [3]> qp_omega;
    Kokkos::View<double* [3]> qp_omega_dot;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        for (auto i = 0U; i < 3U; ++i) {
            qp_x0(i_elem, i) = node_x0(i_elem, i);
            qp_u(i_elem, i) = node_u(i_elem, i);
            qp_u_ddot(i_elem, i) = node_u_ddot(i_elem, i);
            qp_omega(i_elem, i) = node_u_dot(i_elem, i + 3);
            qp_omega_dot(i_elem, i) = node_u_ddot(i_elem, i + 3);
        }
        for (auto i = 0U; i < 4U; ++i) {
            qp_r0(i_elem, i) = node_x0(i_elem, i + 3);
            qp_r(i_elem, i) = node_u(i_elem, i + 3);
        }
    }
};

}  // namespace openturbine::masses
