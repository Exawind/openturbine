#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/calculate_temporary_variables.hpp"
#include "test_calculate.hpp"

namespace {

void TestCalculateTemporaryVariables() {
    const auto x0_prime = Kokkos::View<double[3]>("x0_prime");
    constexpr auto x0_prime_data = std::array{1., 2., 3.};
    const auto x0_prime_host =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(x0_prime_data.data());
    const auto x0_prime_mirror = Kokkos::create_mirror(x0_prime);
    Kokkos::deep_copy(x0_prime_mirror, x0_prime_host);
    Kokkos::deep_copy(x0_prime, x0_prime_mirror);

    const auto u_prime = Kokkos::View<double[3]>("u_prime");
    constexpr auto u_prime_data = std::array{4., 5., 6.};
    const auto u_prime_host =
        Kokkos::View<double[3], Kokkos::HostSpace>::const_type(u_prime_data.data());
    const auto u_prime_mirror = Kokkos::create_mirror(u_prime);
    Kokkos::deep_copy(u_prime_mirror, u_prime_host);
    Kokkos::deep_copy(u_prime, u_prime_mirror);

    const auto x0pupSS = Kokkos::View<double[3][3]>("x0pupSS");

    Kokkos::parallel_for(
        "CalculateTemporaryVariables", 1,
        KOKKOS_LAMBDA(size_t) {
            openturbine::beams::CalculateTemporaryVariables(x0_prime, u_prime, x0pupSS);
        }
    );

    constexpr auto x0pupSS_exact_data = std::array{0., -9., 7., 9., 0., -5., -7., 5., 0.};
    const auto x0pupSS_exact =
        Kokkos::View<double[3][3], Kokkos::HostSpace>::const_type(x0pupSS_exact_data.data());

    const auto x0pupSS_mirror = Kokkos::create_mirror(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, x0pupSS);
    openturbine::tests::CompareWithExpected(x0pupSS_mirror, x0pupSS_exact);
}

}  // namespace

namespace openturbine::tests {

TEST(CalculateTemporaryVariablesTests, OneNode) {
    TestCalculateTemporaryVariables();
}

}  // namespace openturbine::tests
