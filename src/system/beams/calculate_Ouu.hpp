#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

KOKKOS_FUNCTION
inline void CalculateOuu(
    const Kokkos::View<double[6][6]>::const_type& Cuu,
    const Kokkos::View<double[3][3]>::const_type& x0pupSS,
    const Kokkos::View<double[3][3]>::const_type& M_tilde,
    const Kokkos::View<double[3][3]>::const_type& N_tilde, const Kokkos::View<double[6][6]>& Ouu
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Default = KokkosBatched::Algo::Gemm::Default;
    using Gemm = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, Default>;
    auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    auto C21 = Kokkos::subview(Cuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
    KokkosBlas::SerialSet::invoke(0., Ouu);
    auto Ouu_12 = Kokkos::subview(Ouu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    KokkosBlas::serial_axpy(1., N_tilde, Ouu_12);
    Gemm::invoke(1., C11, x0pupSS, -1., Ouu_12);
    auto Ouu_22 = Kokkos::subview(Ouu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    KokkosBlas::serial_axpy(1., M_tilde, Ouu_22);
    Gemm::invoke(1., C21, x0pupSS, -1., Ouu_22);
}

}  // namespace openturbine::beams
