#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::masses {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateInertialForce(
    double mass, const typename Kokkos::View<double[3], DeviceType>::const_type& u_ddot,
    const typename Kokkos::View<double[3], DeviceType>::const_type& omega,
    const typename Kokkos::View<double[3], DeviceType>::const_type& omega_dot,
    const typename Kokkos::View<double[3], DeviceType>::const_type& eta,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& eta_tilde,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& rho,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& omega_tilde,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& omega_dot_tilde,
    const Kokkos::View<double[6], DeviceType>& FI
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using GemmDefault = KokkosBatched::Algo::Gemm::Default;
    using GemvDefault = KokkosBlas::Algo::Gemv::Default;
    using Gemm = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, GemmDefault>;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, GemvDefault>;
    auto v1 = Kokkos::Array<double, 3>{};
    auto V1 = Kokkos::View<double[3], DeviceType>(v1.data());
    auto m1 = Kokkos::Array<double, 9>{};
    auto M1 = Kokkos::View<double[3][3], DeviceType>(m1.data());

    // Compute first 3 components of FI
    // FI_1 = m * u_ddot + (omega_dot_tilde + omega_tilde * omega_tilde) * m * eta
    auto FI_1 = Kokkos::subview(FI, Kokkos::make_pair(0, 3));
    Gemm::invoke(mass, omega_tilde, omega_tilde, 0., M1);
    KokkosBlas::serial_axpy(mass, omega_dot_tilde, M1);
    Gemv::invoke(1., M1, eta, 0., FI_1);
    KokkosBlas::serial_axpy(mass, u_ddot, FI_1);

    // Compute last 3 components of FI
    // FI_2 = m * eta_tilde * u_ddot + rho * omega_dot + omega_tilde * rho * omega
    auto FI_2 = Kokkos::subview(FI, Kokkos::make_pair(3, 6));
    KokkosBlas::serial_axpy(mass, u_ddot, V1);
    Gemv::invoke(1., eta_tilde, V1, 0., FI_2);
    Gemv::invoke(1., rho, omega_dot, 1., FI_2);
    Gemm::invoke(1., omega_tilde, rho, 0., M1);
    Gemv::invoke(1., M1, omega, 1., FI_2);
}

}  // namespace openturbine::masses
