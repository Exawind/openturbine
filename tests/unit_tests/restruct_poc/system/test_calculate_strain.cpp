#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/calculate_strain.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(CalculateStrainTests, OneNode) {
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

    auto r = Kokkos::View<double[1][4]>("r");
    auto r_data = std::array<double, 4>{7., 8., 9., 10.};
    auto r_host = Kokkos::View<double[1][4], Kokkos::HostSpace>(r_data.data());
    auto r_mirror = Kokkos::create_mirror(r);
    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(r, r_mirror);

    auto r_prime = Kokkos::View<double[1][4]>("r_prime");
    auto r_prime_data = std::array<double, 4>{11., 12., 13., 14.};
    auto r_prime_host = Kokkos::View<double[1][4], Kokkos::HostSpace>(r_prime_data.data());
    auto r_prime_mirror = Kokkos::create_mirror(r_prime);
    Kokkos::deep_copy(r_prime_mirror, r_prime_host);
    Kokkos::deep_copy(r_prime, r_prime_mirror);

    auto strain = Kokkos::View<double[1][6]>("strain");

    Kokkos::parallel_for("CalculateStrain", 1, CalculateStrain{x0_prime, u_prime, r, r_prime, strain});

    auto strain_exact_data = std::array<double, 6>{-793., -413., -621., -16., 0., -32.};
    auto strain_exact = Kokkos::View<double[1][6], Kokkos::HostSpace>(strain_exact_data.data());
    
    auto strain_mirror = Kokkos::create_mirror(strain);
    Kokkos::deep_copy(strain_mirror, strain);
    for(int i = 0; i < 6; ++i) {
        EXPECT_EQ(strain_mirror(0, i), strain_exact(0, i));
    }
}

}  // namespace openturbine::restruct_poc::tests