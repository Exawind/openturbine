#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/calculate_force_FC.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

TEST(CalculateForceFCTests, OneNode) {
    const auto Cuu = Kokkos::View<double[1][1][6][6]>("Cuu");
    constexpr auto Cuu_data = std::array{
        1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14., 15., 16., 17., 18.,
        19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
    };
    const auto Cuu_host = Kokkos::View<const double[1][1][6][6], Kokkos::HostSpace>(Cuu_data.data());
    const auto Cuu_mirror = Kokkos::create_mirror(Cuu);
    Kokkos::deep_copy(Cuu_mirror, Cuu_host);
    Kokkos::deep_copy(Cuu, Cuu_mirror);

    const auto strain = Kokkos::View<double[1][1][6]>("strain");
    constexpr auto strain_data = std::array{37., 38., 39., 40., 41., 42.};
    const auto strain_host =
        Kokkos::View<const double[1][1][6], Kokkos::HostSpace>(strain_data.data());
    const auto strain_mirror = Kokkos::create_mirror(strain);
    Kokkos::deep_copy(strain_mirror, strain_host);
    Kokkos::deep_copy(strain, strain_mirror);

    const auto FC = Kokkos::View<double[1][1][6]>("FC");
    const auto M_tilde = Kokkos::View<double[1][1][3][3]>("M_tilde");
    const auto N_tilde = Kokkos::View<double[1][1][3][3]>("N_tilde");

    Kokkos::parallel_for(
        "CalculateForceFC", 1, CalculateForceFC{0, Cuu, strain, FC, M_tilde, N_tilde}
    );

    constexpr auto FC_exact_data = std::array{847., 2269., 3691., 5113., 6535., 7957.};
    const auto FC_exact =
        Kokkos::View<const double[1][1][6], Kokkos::HostSpace>(FC_exact_data.data());

    const auto FC_mirror = Kokkos::create_mirror(FC);
    Kokkos::deep_copy(FC_mirror, FC);
    CompareWithExpected(FC_mirror, FC_exact);

    constexpr auto M_tilde_exact_data =
        std::array{0., -7957., 6535., 7957., 0., -5113., -6535., 5113., 0.};
    const auto M_tilde_exact =
        Kokkos::View<const double[1][1][3][3], Kokkos::HostSpace>(M_tilde_exact_data.data());

    const auto M_tilde_mirror = Kokkos::create_mirror(M_tilde);
    Kokkos::deep_copy(M_tilde_mirror, M_tilde);
    CompareWithExpected(M_tilde_mirror, M_tilde_exact);

    constexpr auto N_tilde_exact_data =
        std::array{0., -3691., 2269., 3691., 0., -847., -2269., 847., 0.};
    const auto N_tilde_exact =
        Kokkos::View<const double[1][1][3][3], Kokkos::HostSpace>(N_tilde_exact_data.data());

    const auto N_tilde_mirror = Kokkos::create_mirror(N_tilde);
    Kokkos::deep_copy(N_tilde_mirror, N_tilde);
    CompareWithExpected(N_tilde_mirror, N_tilde_exact);
}

}  // namespace openturbine::tests
