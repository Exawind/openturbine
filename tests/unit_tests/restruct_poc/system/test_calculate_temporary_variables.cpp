#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_temporary_variables.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateTemporaryVariablesTests, OneNode) {
    auto x0_prime = Kokkos::View<double[1][3]>("x0_prime");
    auto x0_prime_data = std::array<double, 3>{1., 2., 3.};
    auto x0_prime_host = Kokkos::View<double[1][3], Kokkos::HostSpace>(x0_prime_data.data());
    auto x0_prime_mirror = Kokkos::create_mirror(x0_prime);
    Kokkos::deep_copy(x0_prime_mirror, x0_prime_host);
    Kokkos::deep_copy(x0_prime, x0_prime_mirror);

    auto u_prime = Kokkos::View<double[1][3]>("u_prime");
    auto u_prime_data = std::array<double, 3>{4., 5., 6.};
    auto u_prime_host = Kokkos::View<double[1][3], Kokkos::HostSpace>(u_prime_data.data());
    auto u_prime_mirror = Kokkos::create_mirror(u_prime);
    Kokkos::deep_copy(u_prime_mirror, u_prime_host);
    Kokkos::deep_copy(u_prime, u_prime_mirror);

    auto x0pupSS = Kokkos::View<double[1][3][3]>("x0pupSS");

    Kokkos::parallel_for(
        "CalculateTemporaryVariables", 1, CalculateTemporaryVariables{x0_prime, u_prime, x0pupSS}
    );

    auto x0pupSS_exact_data = std::array<double, 9>{0., -9., 7., 9., 0., -5., -7., 5., 0.};
    auto x0pupSS_exact = Kokkos::View<double[1][3][3], Kokkos::HostSpace>(x0pupSS_exact_data.data());

    auto x0pupSS_mirror = Kokkos::create_mirror(x0pupSS);
    Kokkos::deep_copy(x0pupSS_mirror, x0pupSS);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(x0pupSS_mirror(0, i, j), x0pupSS_exact(0, i, j));
        }
    }
}

}  // namespace openturbine::restruct_poc::tests