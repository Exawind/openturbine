#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateForceFC {
    using NoTranspose = KokkosBlas::Trans::NoTranspose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;
    size_t i_elem;
    Kokkos::View<double** [6][6]>::const_type qp_Cuu_;
    Kokkos::View<double** [6]>::const_type qp_strain_;
    Kokkos::View<double** [6]> qp_FC_;
    Kokkos::View<double** [3][3]> M_tilde_;
    Kokkos::View<double** [3][3]> N_tilde_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto strain = Kokkos::subview(qp_strain_, i_elem, i_qp, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_elem, i_qp, Kokkos::ALL);
        auto M_tilde = Kokkos::subview(M_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);

        Gemv::invoke(1., Cuu, strain, 0., FC);
        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        auto M = Kokkos::subview(FC, Kokkos::make_pair(3, 6));
        VecTilde(M, M_tilde);
        VecTilde(N, N_tilde);
    }
};

}  // namespace openturbine
