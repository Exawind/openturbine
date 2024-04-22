#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>

#include "VectorOperations.hpp"
#include "types.hpp"

namespace openturbine {

struct CalculateForceFC {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx6::const_type qp_strain_;
    View_Nx6 qp_FC_;
    View_Nx3x3 M_tilde_;
    View_Nx3x3 N_tilde_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto strain = Kokkos::subview(qp_strain_, i_qp, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_qp, Kokkos::ALL);
        auto M_tilde = Kokkos::subview(M_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);

        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::invoke(1., Cuu, strain, 0., FC);
        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        auto M = Kokkos::subview(FC, Kokkos::make_pair(3, 6));
        VecTilde(M, M_tilde);
        VecTilde(N, N_tilde);
    }
};

}  // namespace openturbine
