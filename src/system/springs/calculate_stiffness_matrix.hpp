#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void CalculateStiffnessMatrix(
    double c1,
    double c2,
    const typename Kokkos::View<double[3], DeviceType>::const_type& r,
    double l,
    const Kokkos::View<double[3][3], DeviceType>& a
) {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using GemmDefault = KokkosBatched::Algo::Gemm::Default;
    using Gemm = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, GemmDefault>;
    auto r_tilde_data = Kokkos::Array<double, 9>{};
    auto r_tilde = Kokkos::View<double[3][3], DeviceType>(r_tilde_data.data());

    const double diag_term = c1 - c2 * l * l;
    a(0, 0) = diag_term;
    a(1, 1) = diag_term;
    a(2, 2) = diag_term;

    VecTilde(r, r_tilde);
    Gemm::invoke(-c2, r_tilde, r_tilde, 1., a);
}

}  // namespace openturbine::springs
