#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_set.hpp>
#include <Kokkos_Core.hpp>

#include "src/restruct_poc/VectorOperations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateGyroscopicMatrix {
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3::const_type qp_omega_;
    View_Nx3x3::const_type omega_tilde_;
    View_Nx3x3::const_type rho_;
    View_Nx3::const_type eta_;
    View_Nx6x6 qp_Guu_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_qp, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_qp, Kokkos::ALL);
        auto v1 = Kokkos::Array<double, 3>{};
        auto V1 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v1.data());
        auto v2 = Kokkos::Array<double, 3>{};
        auto V2 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v2.data());
        auto m1 = Kokkos::Array<double, 9>{};
        auto M1 = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(m1.data());
        auto Guu = Kokkos::subview(qp_Guu_, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);
        // Inertia gyroscopic matrix
        KokkosBlas::SerialSet::invoke(0., Guu);
        // omega.tilde() * m * eta.tilde().t() + (omega.tilde() * m * eta).tilde().t()
        auto Guu_12 = Kokkos::subview(Guu, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
        KokkosBlas::serial_axpy(m, eta, V1);
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::
            invoke(1., omega_tilde, V1, 0., V2);
        VecTilde(V2, M1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Guu_12(i, j) = M1(j, i);
            }
        }

        VecTilde(V1, M1);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::Transpose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., omega_tilde, M1, 1., Guu_12);
        // Guu_22 = omega.tilde() * rho - (rho * omega).tilde()
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Default>::
            invoke(1., rho, omega, 0., V1);
        VecTilde(V1, M1);
        auto Guu_22 = Kokkos::subview(Guu, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Guu_22(i, j) = M1(i, j);
            }
        }
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., omega_tilde, rho, -1., Guu_22);
    }
};

}  // namespace openturbine
