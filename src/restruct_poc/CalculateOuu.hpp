#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "MatrixOperations.hpp"

namespace openturbine {

struct CalculateOuu {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx3x3::const_type x0pupSS_;
    View_Nx3x3::const_type M_tilde_;
    View_Nx3x3::const_type N_tilde_;
    View_Nx6x6 qp_Ouu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M_tilde = Kokkos::subview(M_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Ouu = Kokkos::subview(qp_Ouu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto C21 = Kokkos::subview(Cuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
        for (int i = 0; i < Ouu.extent_int(0); ++i) {
            for (int j = 0; j < Ouu.extent_int(1); ++j) {
                Ouu(i, j) = 0.;
            }
        }
        auto Ouu_12 = Kokkos::subview(Ouu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        auto Ouu_22 = Kokkos::subview(Ouu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(C11, x0pupSS, Ouu_12);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Ouu_12(i, j) -= N_tilde(i, j);
            }
        }
        MatMulAB(C21, x0pupSS, Ouu_22);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Ouu_22(i, j) -= M_tilde(i, j);
            }
        }
    }
};

}
