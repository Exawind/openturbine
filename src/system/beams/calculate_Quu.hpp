#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateQuu(
    const typename Kokkos::View<double[6][6], DeviceType>::const_type& Cuu,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& x0pupSS,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& N_tilde,
    const Kokkos::View<double[6][6], DeviceType>& Quu
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Transpose = KokkosBatched::Trans::Transpose;
    using Default = KokkosBatched::Algo::Gemm::Default;
    using GemmNN = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, Default>;
    using GemmTN = KokkosBatched::SerialGemm<Transpose, NoTranspose, Default>;
    auto m1 = Kokkos::Array<double, 9>{};
    auto M1 = Kokkos::View<double[3][3]>(m1.data());
    auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    KokkosBlas::SerialSet::invoke(0., Quu);
    KokkosBlas::serial_axpy(1., N_tilde, M1);
    GemmNN::invoke(1., C11, x0pupSS, -1., M1);
    auto Quu_22 = Kokkos::subview(Quu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    GemmTN::invoke(1., x0pupSS, M1, 0., Quu_22);
}

}  // namespace openturbine::beams
