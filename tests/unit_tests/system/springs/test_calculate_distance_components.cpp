#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/system/springs/calculate_distance_components.hpp"
#include "tests/unit_tests/system/beams/test_calculate.hpp"

namespace openturbine::tests {

TEST(CalculateDistanceComponentsTests, OneElement) {
    const auto x0 = Kokkos::View<double[1][3]>("x0");
    const auto u1 = Kokkos::View<double[1][3]>("u1");
    const auto u2 = Kokkos::View<double[1][3]>("u2");
    const auto r = Kokkos::View<double[1][3]>("r");

    constexpr auto x0_data = std::array{1., 2., 3.};
    constexpr auto u1_data = std::array{0.1, 0.2, 0.3};
    constexpr auto u2_data = std::array{0.4, 0.5, 0.6};

    const auto x0_host = Kokkos::View<const double[1][3], Kokkos::HostSpace>(x0_data.data());
    const auto u1_host = Kokkos::View<const double[1][3], Kokkos::HostSpace>(u1_data.data());
    const auto u2_host = Kokkos::View<const double[1][3], Kokkos::HostSpace>(u2_data.data());

    const auto x0_mirror = Kokkos::create_mirror(x0);
    const auto u1_mirror = Kokkos::create_mirror(u1);
    const auto u2_mirror = Kokkos::create_mirror(u2);

    Kokkos::deep_copy(x0_mirror, x0_host);
    Kokkos::deep_copy(u1_mirror, u1_host);
    Kokkos::deep_copy(u2_mirror, u2_host);

    Kokkos::deep_copy(x0, x0_mirror);
    Kokkos::deep_copy(u1, u1_mirror);
    Kokkos::deep_copy(u2, u2_mirror);

    Kokkos::parallel_for(
        "CalculateDistanceComponents", 1, CalculateDistanceComponents{x0, u1, u2, r}
    );

    constexpr auto r_exact_data = std::array{
        1. - 0.1 + 0.4,  // 1.3
        2. - 0.2 + 0.5,  // 2.3
        3. - 0.3 + 0.6   // 3.3
    };
    const auto r_exact = Kokkos::View<const double[1][3], Kokkos::HostSpace>(r_exact_data.data());

    const auto r_mirror = Kokkos::create_mirror(r);
    Kokkos::deep_copy(r_mirror, r);

    CompareWithExpected(r_mirror, r_exact);
}

}  // namespace openturbine::tests
