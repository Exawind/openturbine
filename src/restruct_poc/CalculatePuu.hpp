#pragma once

#include <Kokkos_Core.hpp>

#include "MatrixOperations.hpp"
#include "types.hpp"

namespace openturbine {

struct CalculatePuu {
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx3x3::const_type x0pupSS_;
    View_Nx3x3::const_type N_tilde_;
    View_Nx6x6 qp_Puu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto N_tilde = Kokkos::subview(N_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Puu = Kokkos::subview(qp_Puu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto C11 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
        auto C12 = Kokkos::subview(Cuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        for (int i = 0; i < Puu.extent_int(0); ++i) {
            for (int j = 0; j < Puu.extent_int(1); ++j) {
                Puu(i, j) = 0.;
            }
        }
        auto Puu_21 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
        MatMulATB(x0pupSS, C11, Puu_21);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Puu_21(i, j) += N_tilde(i, j);
            }
        }
        auto Puu_22 = Kokkos::subview(Puu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulATB(x0pupSS, C12, Puu_22);
    }
};

}  // namespace openturbine
