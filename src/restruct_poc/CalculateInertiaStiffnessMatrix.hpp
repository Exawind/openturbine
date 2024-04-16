#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "MatrixOperations.hpp"
#include "VectorOperations.hpp"

namespace openturbine {

struct CalculateInertiaStiffnessMatrix {
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3::const_type qp_u_ddot_;
    View_Nx3::const_type qp_omega_;
    View_Nx3::const_type qp_omega_dot_;
    View_Nx3x3::const_type omega_tilde_;
    View_Nx3x3::const_type omega_dot_tilde_;
    View_Nx3x3::const_type rho_;
    View_Nx3::const_type eta_;
    View_Nx3 v1_;
    View_Nx3x3 M1_;
    View_Nx3x3 M2_;
    View_Nx6x6 qp_Kuu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto u_ddot = Kokkos::subview(qp_u_ddot_, i_qp, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_dot = Kokkos::subview(qp_omega_dot_, i_qp, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_dot_tilde = Kokkos::subview(omega_dot_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto M2 = Kokkos::subview(M2_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Kuu = Kokkos::subview(qp_Kuu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);

        for (int i = 0; i < Kuu.extent_int(0); ++i) {
            for (int j = 0; j < Kuu.extent_int(1); ++j) {
                Kuu(i, j) = 0.;
            }
        }
        auto Kuu_12 = Kokkos::subview(Kuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        MatMulAB(omega_tilde, omega_tilde, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M1(i, j) += omega_dot_tilde(i, j);
            }
        }
        VecScale(eta, m, V1);
        VecTilde(V1, M2);
        MatMulABT(M1, M2, Kuu_12);
        auto Kuu_22 = Kokkos::subview(Kuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        VecTilde(u_ddot, M1);
        VecScale(eta, m, V1);
        VecTilde(V1, M2);
        MatMulAB(M1, M2, Kuu_22);
        MatMulAB(rho, omega_dot_tilde, M1);
        MatVecMulAB(rho, omega_dot, V1);
        VecTilde(V1, M2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Kuu_22(i, j) += M1(i, j) - M2(i, j);
            }
        }
        MatMulAB(rho, omega_tilde, M1);
        MatVecMulAB(rho, omega, V1);
        VecTilde(V1, M2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M1(i, j) -= M2(i, j);
            }
        }
        MatMulAB(omega_tilde, M1, M2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Kuu_22(i, j) += M2(i, j);
            }
        }
    }
};

}
