#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "src/restruct_poc/system/calculate_gyroscopic_matrix.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::tests {

TEST(CalculateGyroscopicMatrixTests, OneNode) {
    const auto Muu = Kokkos::View<double[1][6][6]>("Muu");
    constexpr auto Muu_data = std::array{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                         13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
                                         25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.};
    const auto Muu_host = Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(Muu_data.data());
    const auto Muu_mirror = Kokkos::create_mirror(Muu);
    Kokkos::deep_copy(Muu_mirror, Muu_host);
    Kokkos::deep_copy(Muu, Muu_mirror);

    const auto omega = Kokkos::View<double[1][3]>("omega");
    constexpr auto omega_data = std::array{40., 41., 42.};
    const auto omega_host = Kokkos::View<const double[1][3], Kokkos::HostSpace>(omega_data.data());
    const auto omega_mirror = Kokkos::create_mirror(omega);
    Kokkos::deep_copy(omega_mirror, omega_host);
    Kokkos::deep_copy(omega, omega_mirror);

    const auto omega_tilde = Kokkos::View<double[1][3][3]>("omega_tilde");
    constexpr auto omega_tilde_data = std::array{46., 47., 48., 49., 50., 51., 52., 53., 54.};
    const auto omega_tilde_host =
        Kokkos::View<const double[1][3][3], Kokkos::HostSpace>(omega_tilde_data.data());
    const auto omega_tilde_mirror = Kokkos::create_mirror(omega_tilde);
    Kokkos::deep_copy(omega_tilde_mirror, omega_tilde_host);
    Kokkos::deep_copy(omega_tilde, omega_tilde_mirror);

    const auto rho = Kokkos::View<double[1][3][3]>("rho");
    constexpr auto rho_data = std::array{55., 56., 57., 58., 59., 60., 61., 62., 63.};
    const auto rho_host = Kokkos::View<const double[1][3][3], Kokkos::HostSpace>(rho_data.data());
    const auto rho_mirror = Kokkos::create_mirror(rho);
    Kokkos::deep_copy(rho_mirror, rho_host);
    Kokkos::deep_copy(rho, rho_mirror);

    const auto eta = Kokkos::View<double[1][3]>("eta");
    constexpr auto eta_data = std::array{64., 65., 66.};
    const auto eta_host = Kokkos::View<const double[1][3], Kokkos::HostSpace>(eta_data.data());
    const auto eta_mirror = Kokkos::create_mirror(eta);
    Kokkos::deep_copy(eta_mirror, eta_host);
    Kokkos::deep_copy(eta, eta_mirror);

    const auto Guu = Kokkos::View<double[1][6][6]>("Guu");

    Kokkos::parallel_for(
        "CalculateGyroscopicMatrix", 1,
        CalculateGyroscopicMatrix{Muu, omega, omega_tilde, rho, eta, Guu}
    );

    constexpr auto Guu_exact_data =
        std::array{0., 0., 0., 18.,   10301., -9734., 0., 0., 0., -10322., -30.,   9182.,
                   0., 0., 0., 9764., -9191., 12.,    0., 0., 0., 8184.,   15953., 1207.,
                   0., 0., 0., 1078., 8856.,  15896., 0., 0., 0., 16487.,  2497.,  9546.};
    const auto Guu_exact =
        Kokkos::View<const double[1][6][6], Kokkos::HostSpace>(Guu_exact_data.data());

    const auto Guu_mirror = Kokkos::create_mirror(Guu);
    Kokkos::deep_copy(Guu_mirror, Guu);
    CompareWithExpected(Guu_mirror, Guu_exact);
}

}  // namespace openturbine::tests