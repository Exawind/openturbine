#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/restruct_poc/math/VectorOperations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateGravityForce {
    View_3::const_type gravity;
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3x3::const_type eta_tilde_;
    View_Nx6 qp_FG_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FG = Kokkos::subview(qp_FG_, i_qp, Kokkos::ALL);
        auto m = Muu(0, 0);
        for (int i = 0; i < 3; ++i) {
            FG(i) = m * gravity(i);
        }
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::
            invoke(
                1., eta_tilde, Kokkos::subview(FG, Kokkos::make_pair(0, 3)), 0.,
                Kokkos::subview(FG, Kokkos::make_pair(3, 6))
            );
    }
};

}  // namespace openturbine
