#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "src/system/beams/calculate_strain.hpp"

namespace openturbine::tests {

TEST(CalculateStrainTests, OneNode) {
    const auto x0_prime = Kokkos::View<double[1][1][3]>("x0_prime");
    constexpr auto x0_prime_data = std::array{1., 2., 3.};
    const auto x0_prime_host =
        Kokkos::View<const double[1][1][3], Kokkos::HostSpace>(x0_prime_data.data());
    const auto x0_prime_mirror = Kokkos::create_mirror(x0_prime);
    Kokkos::deep_copy(x0_prime_mirror, x0_prime_host);
    Kokkos::deep_copy(x0_prime, x0_prime_mirror);

    const auto u_prime = Kokkos::View<double[1][1][3]>("u_prime");
    constexpr auto u_prime_data = std::array{4., 5., 6.};
    const auto u_prime_host =
        Kokkos::View<const double[1][1][3], Kokkos::HostSpace>(u_prime_data.data());
    const auto u_prime_mirror = Kokkos::create_mirror(u_prime);
    Kokkos::deep_copy(u_prime_mirror, u_prime_host);
    Kokkos::deep_copy(u_prime, u_prime_mirror);

    const auto r = Kokkos::View<double[1][1][4]>("r");
    constexpr auto r_data = std::array<double, 4>{7., 8., 9., 10.};
    const auto r_host = Kokkos::View<const double[1][1][4], Kokkos::HostSpace>(r_data.data());
    const auto r_mirror = Kokkos::create_mirror(r);
    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(r, r_mirror);

    const auto r_prime = Kokkos::View<double[1][1][4]>("r_prime");
    constexpr auto r_prime_data = std::array{11., 12., 13., 14.};
    const auto r_prime_host =
        Kokkos::View<const double[1][1][4], Kokkos::HostSpace>(r_prime_data.data());
    const auto r_prime_mirror = Kokkos::create_mirror(r_prime);
    Kokkos::deep_copy(r_prime_mirror, r_prime_host);
    Kokkos::deep_copy(r_prime, r_prime_mirror);

    const auto strain = Kokkos::View<double[1][1][6]>("strain");

    Kokkos::parallel_for(
        "CalculateStrain", 1, CalculateStrain{0, x0_prime, u_prime, r, r_prime, strain}
    );

    constexpr auto strain_exact_data = std::array{-793., -413., -621., -16., 0., -32.};
    const auto strain_exact =
        Kokkos::View<const double[1][1][6], Kokkos::HostSpace>(strain_exact_data.data());

    const auto strain_mirror = Kokkos::create_mirror(strain);
    Kokkos::deep_copy(strain_mirror, strain);
    CompareWithExpected(strain_mirror, strain_exact);
}

}  // namespace openturbine::tests
