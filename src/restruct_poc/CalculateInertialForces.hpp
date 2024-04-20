#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>

#include "MatrixOperations.hpp"
#include "VectorOperations.hpp"
#include "types.hpp"

namespace openturbine {

struct CalculateInertialForces {
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3::const_type qp_u_ddot_;
    View_Nx3::const_type qp_omega_;
    View_Nx3::const_type qp_omega_dot_;
    View_Nx3x3::const_type eta_tilde_;
    View_Nx3x3 omega_tilde_;
    View_Nx3x3 omega_dot_tilde_;
    View_Nx3x3::const_type rho_;
    View_Nx3::const_type eta_;
    View_Nx3 v1_;
    View_Nx3x3 M1_;
    View_Nx6 qp_FI_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto u_ddot = Kokkos::subview(qp_u_ddot_, i_qp, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_dot = Kokkos::subview(qp_omega_dot_, i_qp, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_dot_tilde = Kokkos::subview(omega_dot_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto M1 = Kokkos::subview(M1_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FI = Kokkos::subview(qp_FI_, i_qp, Kokkos::ALL);

        auto m = Muu(0, 0);
        VecTilde(omega, omega_tilde);
        VecTilde(omega_dot, omega_dot_tilde);
        auto FI_1 = Kokkos::subview(FI, Kokkos::make_pair(0, 3));
        KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Default>::invoke(1., omega_tilde, omega_tilde, 0., M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M1(i, j) += omega_dot_tilde(i, j);
                M1(i, j) *= m;
            }
        }
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::invoke(1., M1, eta, 0., FI_1);
        for (int i = 0; i < 3; i++) {
            FI_1(i) += u_ddot(i) * m;
        }
        auto FI_2 = Kokkos::subview(FI, Kokkos::make_pair(3, 6));
        VecScale(u_ddot, m, V1);
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::invoke(1., eta_tilde, V1, 0., FI_2);
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::invoke(1., rho, omega_dot, 0., V1);
        for (int i = 0; i < 3; i++) {
            FI_2(i) += V1(i);
        }
        KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Default>::invoke(1., omega_tilde, rho, 0., M1);
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::invoke(1., M1, omega, 0., V1);
        for (int i = 0; i < 3; i++) {
            FI_2(i) += V1(i);
        }
    }
};

}  // namespace openturbine
