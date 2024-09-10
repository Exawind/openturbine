#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateGravityForce {
    using NoTranspose = KokkosBlas::Trans::NoTranspose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;
    size_t i_elem;
    Kokkos::View<double[3]>::const_type gravity;
    Kokkos::View<double** [6][6]>::const_type qp_Muu_;
    Kokkos::View<double** [3][3]>::const_type eta_tilde_;
    Kokkos::View<double** [6]> qp_FG_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FG = Kokkos::subview(qp_FG_, i_elem, i_qp, Kokkos::ALL);
        auto m = Muu(0, 0);
        for (int i = 0; i < 3; ++i) {
            FG(i) = m * gravity(i);
        }
        Gemv::invoke(
            1., eta_tilde, Kokkos::subview(FG, Kokkos::make_pair(0, 3)), 0.,
            Kokkos::subview(FG, Kokkos::make_pair(3, 6))
        );
    }
};

}  // namespace openturbine
