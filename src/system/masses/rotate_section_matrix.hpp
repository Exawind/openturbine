#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine::masses {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void RotateSectionMatrix(
    const typename Kokkos::View<double[4], DeviceType>::const_type& xr,
    const typename Kokkos::View<double[6][6], DeviceType>::const_type& Cstar,
    const Kokkos::View<double[6][6], DeviceType>& Cuu
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Transpose = KokkosBatched::Trans::Transpose;
    using Default = KokkosBatched::Algo::Gemm::Default;
    using GemmNN = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, Default>;
    using GemmNT = KokkosBatched::SerialGemm<NoTranspose, Transpose, Default>;

    auto RR0_data = Kokkos::Array<double, 9>{};
    auto RR0 = Kokkos::View<double[3][3], DeviceType>(RR0_data.data());
    QuaternionToRotationMatrix(xr, RR0);

    const auto Cstar_top = Kokkos::subview(Cstar, Kokkos::make_pair(0, 3), Kokkos::ALL);
    const auto Cstar_bottom = Kokkos::subview(Cstar, Kokkos::make_pair(3, 6), Kokkos::ALL);
    const auto Cuu_left = Kokkos::subview(Cuu, Kokkos::ALL, Kokkos::make_pair(0, 3));
    const auto Cuu_right = Kokkos::subview(Cuu, Kokkos::ALL, Kokkos::make_pair(3, 6));

    auto ctmp_data = Kokkos::Array<double, 36>{};
    const auto Ctmp = Kokkos::View<double[6][6], DeviceType>(ctmp_data.data());
    const auto Ctmp_top = Kokkos::subview(Ctmp, Kokkos::make_pair(0, 3), Kokkos::ALL);
    const auto Ctmp_bottom = Kokkos::subview(Ctmp, Kokkos::make_pair(3, 6), Kokkos::ALL);
    const auto Ctmp_left = Kokkos::subview(Ctmp, Kokkos::ALL, Kokkos::make_pair(0, 3));
    const auto Ctmp_right = Kokkos::subview(Ctmp, Kokkos::ALL, Kokkos::make_pair(3, 6));

    GemmNN::invoke(1., RR0, Cstar_top, 0., Ctmp_top);
    GemmNN::invoke(1., RR0, Cstar_bottom, 0., Ctmp_bottom);
    GemmNT::invoke(1., Ctmp_left, RR0, 0., Cuu_left);
    GemmNT::invoke(1., Ctmp_right, RR0, 0., Cuu_right);
}

}  // namespace openturbine::masses
