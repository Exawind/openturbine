#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_inertial_forces.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateInertialForcesTests, OneNode) {
    auto Muu = Kokkos::View<double[1][6][6]>("Muu");
    auto Muu_data = std::array<double, 36>{
        1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14., 15., 16., 17., 18.,
        19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    auto Muu_host = Kokkos::View<double[1][6][6], Kokkos::HostSpace>(Muu_data.data());
    auto Muu_mirror = Kokkos::create_mirror(Muu);
    Kokkos::deep_copy(Muu_mirror, Muu_host);
    Kokkos::deep_copy(Muu, Muu_mirror);

    auto u_ddot = Kokkos::View<double[1][3]>("u_ddot");
    auto u_ddot_data = std::array<double, 3>{37., 38., 39.};
    auto u_ddot_host = Kokkos::View<double[1][3], Kokkos::HostSpace>(u_ddot_data.data());
    auto u_ddot_mirror = Kokkos::create_mirror(u_ddot);
    Kokkos::deep_copy(u_ddot_mirror, u_ddot_host);
    Kokkos::deep_copy(u_ddot, u_ddot_mirror);

    auto omega = Kokkos::View<double[1][3]>("omega");
    auto omega_data = std::array<double, 3>{40., 41., 42.};
    auto omega_host = Kokkos::View<double[1][3], Kokkos::HostSpace>(omega_data.data());
    auto omega_mirror = Kokkos::create_mirror(omega);
    Kokkos::deep_copy(omega_mirror, omega_host);
    Kokkos::deep_copy(omega, omega_mirror);

    auto omega_dot = Kokkos::View<double[1][3]>("omega_dot");
    auto omega_dot_data = std::array<double, 3>{43., 44., 45.};
    auto omega_dot_host = Kokkos::View<double[1][3], Kokkos::HostSpace>(omega_dot_data.data());
    auto omega_dot_mirror = Kokkos::create_mirror(omega_dot);
    Kokkos::deep_copy(omega_dot_mirror, omega_dot_host);
    Kokkos::deep_copy(omega_dot, omega_dot_mirror);

    auto eta_tilde = Kokkos::View<double[1][3][3]>("eta_tilde");
    auto eta_tilde_data = std::array<double, 9>{46., 47., 48., 49., 50., 51., 52., 53., 54.};
    auto eta_tilde_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(eta_tilde_data.data());
    auto eta_tilde_mirror = Kokkos::create_mirror(eta_tilde);
    Kokkos::deep_copy(eta_tilde_mirror, eta_tilde_host);
    Kokkos::deep_copy(eta_tilde, eta_tilde_mirror);

    auto rho = Kokkos::View<double[1][3][3]>("rho");
    auto rho_data = std::array<double, 9>{55., 56., 57., 58., 59., 60., 61., 62., 63.};
    auto rho_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(rho_data.data());
    auto rho_mirror = Kokkos::create_mirror(rho);
    Kokkos::deep_copy(rho_mirror, rho_host);
    Kokkos::deep_copy(rho, rho_mirror);

    auto eta = Kokkos::View<double[1][3]>("eta");
    auto eta_data = std::array<double, 3>{64., 65., 66.};
    auto eta_host = Kokkos::View<double[1][3], Kokkos::HostSpace>(eta_data.data());
    auto eta_mirror = Kokkos::create_mirror(eta);
    Kokkos::deep_copy(eta_mirror, eta_host);
    Kokkos::deep_copy(eta, eta_mirror);

    auto omega_tilde = Kokkos::View<double[1][3][3]>("omega_tilde");
    auto omega_dot_tilde = Kokkos::View<double[1][3][3]>("omega_dot_tilde");
    auto FI = Kokkos::View<double[1][6]>("FI");

    Kokkos::parallel_for(
        "CalculateInertialForces", 1,
        CalculateInertialForces{
            Muu, u_ddot, omega, omega_dot, eta_tilde, omega_tilde, omega_dot_tilde, rho, eta, FI}
    );

    auto omega_tilde_exact_data = std::array<double, 9>{0., -42., 41., 42., 0., -40., -41., 40., 0.};
    auto omega_tilde_exact =
        Kokkos::View<double[1][3][3], Kokkos::HostSpace>(omega_tilde_exact_data.data());

    auto omega_tilde_mirror = Kokkos::create_mirror(omega_tilde);
    Kokkos::deep_copy(omega_tilde_mirror, omega_tilde);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(omega_tilde_mirror(0, i, j), omega_tilde_exact(0, i, j));
        }
    }

    auto omega_dot_tilde_exact_data =
        std::array<double, 9>{0., -45., 44., 45., 0., -43., -44., 43., 0.};
    auto omega_dot_tilde_exact =
        Kokkos::View<double[1][3][3], Kokkos::HostSpace>(omega_dot_tilde_exact_data.data());

    auto omega_dot_tilde_mirror = Kokkos::create_mirror(omega_dot_tilde);
    Kokkos::deep_copy(omega_dot_tilde_mirror, omega_dot_tilde);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(omega_dot_tilde_mirror(0, i, j), omega_dot_tilde_exact(0, i, j));
        }
    }

    auto FI_exact_data = std::array<double, 6>{-2984., 32., 2922., 20624., -2248., 22100.};
    auto FI_exact = Kokkos::View<double[1][6], Kokkos::HostSpace>(FI_exact_data.data());

    auto FI_mirror = Kokkos::create_mirror(FI);
    Kokkos::deep_copy(FI_mirror, FI);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(FI_mirror(0, i), FI_exact(0, i));
    }
}

}  // namespace openturbine::restruct_poc::tests