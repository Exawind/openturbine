#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::masses {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void CalculateInertiaStiffnessMatrix(
    double mass,
    const typename Kokkos::View<double[3], DeviceType>::const_type& u_ddot,
    const typename Kokkos::View<double[3], DeviceType>::const_type& omega,
    const typename Kokkos::View<double[3], DeviceType>::const_type& omega_dot,
    const typename Kokkos::View<double[3], DeviceType>::const_type& eta,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& rho,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& omega_tilde,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& omega_dot_tilde,
    const Kokkos::View<double[6][6], DeviceType>& Kuu
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Transpose = KokkosBatched::Trans::Transpose;
    using GemmDefault = KokkosBatched::Algo::Gemm::Default;
    using GemvDefault = KokkosBatched::Algo::Gemv::Default;
    using GemmNN = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, GemmDefault>;
    using GemmNT = KokkosBatched::SerialGemm<NoTranspose, Transpose, GemmDefault>;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, GemvDefault>;

    auto v1 = Kokkos::Array<double, 3>{};
    auto V1 = Kokkos::View<double[3], DeviceType>(v1.data());
    auto m1 = Kokkos::Array<double, 9>{};
    auto M1 = Kokkos::View<double[3][3], DeviceType>(m1.data());
    auto m2 = Kokkos::Array<double, 9>{};
    auto M2 = Kokkos::View<double[3][3], DeviceType>(m2.data());

    KokkosBlas::serial_axpy(1., omega_dot_tilde, M1);
    GemmNN::invoke(1., omega_tilde, omega_tilde, 1., M1);
    KokkosBlas::serial_axpy(mass, eta, V1);
    VecTilde(V1, M2);
    auto Kuu_12 = Kokkos::subview(Kuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    GemmNT::invoke(1., M1, M2, 0., Kuu_12);
    VecTilde(u_ddot, M1);
    auto Kuu_22 = Kokkos::subview(Kuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    GemmNN::invoke(1., rho, omega_dot_tilde, 0., Kuu_22);
    GemmNN::invoke(1., M1, M2, 1., Kuu_22);
    Gemv::invoke(1., rho, omega_dot, 0., V1);
    VecTilde(V1, M2);
    KokkosBlas::serial_axpy(-1., M2, Kuu_22);
    Gemv::invoke(1., rho, omega, 0., V1);
    VecTilde(V1, M1);
    GemmNN::invoke(1., rho, omega_tilde, -1., M1);
    GemmNN::invoke(1., omega_tilde, M1, 1., Kuu_22);
}

}  // namespace openturbine::masses
