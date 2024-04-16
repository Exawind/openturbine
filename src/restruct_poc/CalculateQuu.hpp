#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "MatrixOperations.hpp"

namespace openturbine {

struct CalculateQuu {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx3x3::const_type x0pupSS_;
    View_Nx3x3::const_type N_tilde_;
    View_Nx3x3 M1_;
    View_Nx6x6 qp_Quu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Quu = Kokkos::subview(qp_Quu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        for (int i = 0; i < Quu.extent_int(0); ++i) {
            for (int j = 0; j < Quu.extent_int(1); ++j) {
                Quu(i, j) = 0.;
            }
        }
        auto Quu_22 = Kokkos::subview(Quu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M1(i, j) -= N_tilde(i, j);
            }
        }
        MatMulATB(x0pupSS, M1, Quu_22);
    }
};

}
