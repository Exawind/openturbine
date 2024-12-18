#pragma once

#include <iostream>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateStiffnessMatrix {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using GemmDefault = KokkosBatched::Algo::Gemm::Default;
    using Gemm = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, GemmDefault>;
    Kokkos::View<double*>::const_type c1_;
    Kokkos::View<double*>::const_type c2_;
    Kokkos::View<double* [3]>::const_type r_;
    Kokkos::View<double*>::const_type l_;
    Kokkos::View<double* [3][3]> r_tilde_;
    Kokkos::View<double* [3][3]> a_;

    KOKKOS_FUNCTION
    void operator()(int i_elem) const {
        auto r = Kokkos::subview(r_, i_elem, Kokkos::ALL);
        auto r_tilde = Kokkos::subview(r_tilde_, i_elem, Kokkos::ALL, Kokkos::ALL);
        auto a = Kokkos::subview(a_, i_elem, Kokkos::ALL, Kokkos::ALL);

        // diagonal terms: c1 - c2 * l^2
        const double diag_term = c1_(i_elem) - c2_(i_elem) * l_(i_elem) * l_(i_elem);
        a(0, 0) = diag_term;
        a(1, 1) = diag_term;
        a(2, 2) = diag_term;

        // non-diagonal terms: -c2 * r_tilde * r_tilde
        auto temp = Kokkos::Array<double, 9>{0.};
        auto temp_view = View_3x3(temp.data());
        VecTilde(r, r_tilde);
        Gemm::invoke(1., r_tilde, r_tilde, 0., temp_view);
        KokkosBlas::serial_axpy(-c2_(i_elem), temp_view, a);
    };
};

}  // namespace openturbine
