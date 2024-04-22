#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

#include "VectorOperations.hpp"
#include "types.hpp"

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
        auto v1 = Kokkos::Array<double, 3>{};
        auto V1 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v1.data());
        auto m1 = Kokkos::Array<double, 9>{};
        auto M1 = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(m1.data());
        auto m2 = Kokkos::Array<double, 9>{};
        auto M2 = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(m2.data());
        auto Kuu = Kokkos::subview(qp_Kuu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);

        KokkosBlas::SerialSet::invoke(0., Kuu);
        KokkosBlas::serial_axpy(1., omega_dot_tilde, M1);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., omega_tilde, omega_tilde, 1., M1);
        KokkosBlas::serial_axpy(m, eta, V1);
        VecTilde(V1, M2);
        auto Kuu_12 = Kokkos::subview(Kuu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::Transpose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., M1, M2, 0., Kuu_12);
        VecTilde(u_ddot, M1);
        auto Kuu_22 = Kokkos::subview(Kuu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., rho, omega_dot_tilde, 0., Kuu_22);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., M1, M2, 1., Kuu_22);
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::
            invoke(1., rho, omega_dot, 0., V1);
        VecTilde(V1, M2);
        KokkosBlas::serial_axpy(-1., M2, Kuu_22);
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::
            invoke(1., rho, omega, 0., V1);
        VecTilde(V1, M1);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., rho, omega_tilde, -1., M1);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., omega_tilde, M1, 1., Kuu_22);
    }
};

}  // namespace openturbine
