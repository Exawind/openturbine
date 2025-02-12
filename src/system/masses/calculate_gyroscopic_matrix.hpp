#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::masses {

KOKKOS_FUNCTION
inline void CalculateGyroscopicMatrix(
    double mass, const Kokkos::View<double[3]>::const_type& omega,
    const Kokkos::View<double[3]>::const_type& eta, const Kokkos::View<double[3][3]>::const_type rho,
    const Kokkos::View<double[3][3]>::const_type& omega_tilde, const Kokkos::View<double[6][6]>& Guu
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Transpose = KokkosBatched::Trans::Transpose;
    using GemmDefault = KokkosBatched::Algo::Gemm::Default;
    using GemvDefault = KokkosBatched::Algo::Gemv::Default;
    using GemmNN = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, GemmDefault>;
    using GemmNT = KokkosBatched::SerialGemm<NoTranspose, Transpose, GemmDefault>;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, GemvDefault>;
    auto v1 = Kokkos::Array<double, 3>{};
    auto V1 = Kokkos::View<double[3]>(v1.data());
    auto v2 = Kokkos::Array<double, 3>{};
    auto V2 = Kokkos::View<double[3]>(v2.data());
    auto m1 = Kokkos::Array<double, 9>{};
    auto M1 = Kokkos::View<double[3][3]>(m1.data());

    // omega.tilde() * m * eta.tilde().t() + (omega.tilde() * m * eta).tilde().t()
    auto Guu_12 = Kokkos::subview(Guu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    KokkosBlas::serial_axpy(mass, eta, V1);
    Gemv::invoke(1., omega_tilde, V1, 0., V2);
    VecTilde(V2, M1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Guu_12(i, j) = M1(j, i);
        }
    }

    VecTilde(V1, M1);
    GemmNT::invoke(1., omega_tilde, M1, 1., Guu_12);
    // Guu_22 = omega.tilde() * rho - (rho * omega).tilde()
    Gemv::invoke(1., rho, omega, 0., V1);
    VecTilde(V1, M1);
    auto Guu_22 = Kokkos::subview(Guu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Guu_22(i, j) = M1(i, j);
        }
    }
    GemmNN::invoke(1., omega_tilde, rho, -1., Guu_22);
}

}  // namespace openturbine::masses
