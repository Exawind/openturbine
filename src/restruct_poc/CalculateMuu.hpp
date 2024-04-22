#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

struct CalculateMuu {
    View_Nx6x6::const_type qp_RR0_;    //
    View_Nx6x6::const_type qp_Mstar_;  //
    View_Nx6x6 qp_Muu_;                //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mstar = Kokkos::subview(qp_Mstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto mtmp_data = Kokkos::Array<double, 36>{};
        auto Mtmp =
            Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{mtmp_data.data()};
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., RR0, Mstar, 0., Mtmp);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::Transpose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., Mtmp, RR0, 0., Muu);
    }
};

}  // namespace openturbine
