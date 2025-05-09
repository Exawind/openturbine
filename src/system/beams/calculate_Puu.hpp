#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void CalculatePuu(
    const typename Kokkos::View<double[6][6], DeviceType>::const_type& Cuu,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& x0pupSS,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& N_tilde,
    const Kokkos::View<double[6][6], DeviceType>& Puu
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Transpose = KokkosBatched::Trans::Transpose;
    using Default = KokkosBatched::Algo::Gemm::Default;
    using GemmTN = KokkosBatched::SerialGemm<Transpose, NoTranspose, Default>;
    auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    auto C12 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    KokkosBlas::SerialSet::invoke(0., Puu);
    auto Puu_21 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
    KokkosBlas::serial_axpy(1., N_tilde, Puu_21);
    GemmTN::invoke(1., x0pupSS, C11, 1., Puu_21);
    auto Puu_22 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    GemmTN::invoke(1., x0pupSS, C12, 0., Puu_22);
}

}  // namespace openturbine::beams
