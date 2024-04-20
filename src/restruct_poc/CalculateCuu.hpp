#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>


#include "MatrixOperations.hpp"
#include "types.hpp"

namespace openturbine {

struct CalculateCuu {
    View_Nx6x6::const_type qp_RR0_;    //
    View_Nx6x6::const_type qp_Cstar_;  //
    View_Nx6x6 qp_Cuu_;                //
    View_Nx6x6 qp_Ctmp_;               //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cstar = Kokkos::subview(qp_Cstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto ctmp_data = Kokkos::Array<double, 36>{};
        auto Ctmp = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctmp_data.data());
        KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Default>::invoke(1., RR0, Cstar, 0., Ctmp);
        KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::Transpose, KokkosBatched::Algo::Gemm::Default>::invoke(1., Ctmp, RR0, 0., Cuu);
    }
};

}  // namespace openturbine
