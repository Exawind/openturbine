#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateQuu {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Transpose = KokkosBatched::Trans::Transpose;
    using Default = KokkosBatched::Algo::Gemm::Default;
    using GemmNN = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, Default>;
    using GemmTN = KokkosBatched::SerialGemm<Transpose, NoTranspose, Default>;
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx3x3::const_type x0pupSS_;
    View_Nx3x3::const_type N_tilde_;
    View_Nx6x6 qp_Quu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto m1 = Kokkos::Array<double, 9>{};
        auto M1 = Kokkos::View<double[3][3]>(m1.data());
        auto Quu = Kokkos::subview(qp_Quu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        KokkosBlas::SerialSet::invoke(0., Quu);
        KokkosBlas::serial_axpy(1., N_tilde, M1);
        GemmNN::invoke(1., C11, x0pupSS, -1., M1);
        auto Quu_22 = Kokkos::subview(Quu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        GemmTN::invoke(1., x0pupSS, M1, 0., Quu_22);
    }
};

}  // namespace openturbine
