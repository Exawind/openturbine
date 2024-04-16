#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "MatrixOperations.hpp"
#include "VectorOperations.hpp"

namespace openturbine {

struct CalculateGyroscopicMatrix {
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3::const_type qp_omega_;
    View_Nx3x3::const_type omega_tilde_;
    View_Nx3x3::const_type rho_;
    View_Nx3::const_type eta_;
    View_Nx3 v1_;
    View_Nx3 v2_;
    View_Nx3x3 M1_;
    View_Nx6x6 qp_Guu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto V2 = Kokkos::subview(v2_, i_qp, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Guu = Kokkos::subview(qp_Guu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);
        // Inertia gyroscopic matrix
        for (int i = 0; i < Guu.extent_int(0); ++i) {
            for (int j = 0; j < Guu.extent_int(1); ++j) {
                Guu(i, j) = 0.;
            }
        }
        // omega.tilde() * m * eta.tilde().t() + (omega.tilde() * m * eta).tilde().t()
        auto Guu_12 = Kokkos::subview(Guu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        VecScale(eta, m, V1);
        VecTilde(V1, M1);
        MatMulABT(omega_tilde, M1, Guu_12);
        MatVecMulAB(omega_tilde, V1, V2);
        VecTilde(V2, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Guu_12(i, j) += M1(j, i);
            }
        }
        // Guu_22 = omega.tilde() * rho - (rho * omega).tilde()
        auto Guu_22 = Kokkos::subview(Guu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        MatMulAB(omega_tilde, rho, Guu_22);
        MatVecMulAB(rho, omega, V1);
        VecTilde(V1, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Guu_22(i, j) -= M1(i, j);
            }
        }
    }
};

}
