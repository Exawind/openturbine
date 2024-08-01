#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_calculate.hpp"

#include "src/restruct_poc/system/calculate_temporary_variables.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::tests {

TEST(CalculateTemporaryVariablesTests, OneNode) {
    const auto x0_prime = Kokkos::View<double[1][3]>("x0_prime");
    constexpr auto x0_prime_data = std::array{1., 2., 3.};
    const auto x0_prime_host =
        Kokkos::View<const double[1][3], Kokkos::HostSpace>(x0_prime_data.data());
    const auto x0_prime_mirror = Kokkos::create_mirror(x0_prime);
    Kokkos::deep_copy(x0_prime_mirror, x0_prime_host);
    Kokkos::deep_copy(x0_prime, x0_prime_mirror);

    const auto u_prime = Kokkos::View<double[1][3]>("u_prime");
    constexpr auto u_prime_data = std::array{4., 5., 6.};
    const auto u_prime_host =
        Kokkos::View<const double[1][3], Kokkos::HostSpace>(u_prime_data.data());
    const auto u_prime_mirror = Kokkos::create_mirror(u_prime);
    Kokkos::deep_copy(u_prime_mirror, u_prime_host);
    Kokkos::deep_copy(u_prime, u_prime_mirror);

    const auto x0pupSS = Kokkos::View<double[1][3][3]>("x0pupSS");

    Kokkos::parallel_for(
        "CalculateTemporaryVariables", 1, CalculateTemporaryVariables{x0_prime, u_prime, x0pupSS}
    );

    constexpr auto x0pupSS_exact_data = std::array{0., -9., 7., 9., 0., -5., -7., 5., 0.};
    const auto x0pupSS_exact =
        Kokkos::View<const double[1][3][3], Kokkos::HostSpace>(x0pupSS_exact_data.data());

    const auto x0pupSS_mirror = Kokkos::create_mirror(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, x0pupSS);
    CompareWithExpected(x0pupSS_mirror, x0pupSS_exact);
}

}  // namespace openturbine::tests