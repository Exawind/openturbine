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
    size_t i_elem;
    Kokkos::View<double** [6][6]>::const_type qp_RR0_;
    Kokkos::View<double** [6][6]>::const_type qp_Cstar_;
    Kokkos::View<double** [6][6]> qp_Cuu_;

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        const auto RR0 =
            Kokkos::subview(qp_RR0_, i_elem, i_qp, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        const auto Cstar_top =
            Kokkos::subview(qp_Cstar_, i_elem, i_qp, Kokkos::make_pair(0, 3), Kokkos::ALL);
        const auto Cstar_bottom =
            Kokkos::subview(qp_Cstar_, i_elem, i_qp, Kokkos::make_pair(3, 6), Kokkos::ALL);
        auto ctmp_data = Kokkos::Array<double, 36>{};
        const auto Ctmp = View_6x6(ctmp_data.data());
        const auto Ctmp_top = Kokkos::subview(Ctmp, Kokkos::make_pair(0, 3), Kokkos::ALL);
        const auto Ctmp_bottom = Kokkos::subview(Ctmp, Kokkos::make_pair(3, 6), Kokkos::ALL);
        const auto Ctmp_left = Kokkos::subview(Ctmp, Kokkos::ALL, Kokkos::make_pair(0, 3));
        const auto Ctmp_right = Kokkos::subview(Ctmp, Kokkos::ALL, Kokkos::make_pair(3, 6));
        GemmNN::invoke(1., RR0, Cstar_top, 0., Ctmp_top);
        GemmNN::invoke(1., RR0, Cstar_bottom, 0., Ctmp_bottom);

        const auto Cuu_left =
            Kokkos::subview(qp_Cuu_, i_elem, i_qp, Kokkos::ALL, Kokkos::make_pair(0, 3));
        const auto Cuu_right =
            Kokkos::subview(qp_Cuu_, i_elem, i_qp, Kokkos::ALL, Kokkos::make_pair(3, 6));
        GemmNT::invoke(1., Ctmp_left, RR0, 0., Cuu_left);
        GemmNT::invoke(1., Ctmp_right, RR0, 0., Cuu_right);
    }
};

}  // namespace openturbine
