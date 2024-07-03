#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct RotateSectionMatrix {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using Transpose = KokkosBatched::Trans::Transpose;
    using Default = KokkosBatched::Algo::Gemm::Default;
    using GemmNN = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, Default>;
    using GemmNT = KokkosBatched::SerialGemm<NoTranspose, Transpose, Default>;
    View_Nx6x6::const_type qp_RR0_;
    View_Nx6x6::const_type qp_Cstar_;
    View_Nx6x6 qp_Cuu_;

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cstar = Kokkos::subview(qp_Cstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto ctmp_data = Kokkos::Array<double, 36>{};
        auto Ctmp = Kokkos::View<double[6][6]>(ctmp_data.data());
        GemmNN::invoke(1., RR0, Cstar, 0., Ctmp);
        GemmNT::invoke(1., Ctmp, RR0, 0., Cuu);
    }
};

}  // namespace openturbine
