#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/system/springs/calculate_force_coefficients.hpp"
#include "tests/unit_tests/system/test_calculate.hpp"

namespace openturbine::tests {

TEST(CalculateForceCoefficientsTests, ThreeElements) {
    const auto k = Kokkos::View<double[3]>("k");
    const auto l_ref = Kokkos::View<double[3]>("l_ref");
    const auto l = Kokkos::View<double[3]>("l");
    const auto c1 = Kokkos::View<double[3]>("c1");
    const auto c2 = Kokkos::View<double[3]>("c2");

    constexpr auto k_data = std::array{100., 200., 300.};  // Spring constants
    constexpr auto l_ref_data = std::array{1., 5., 3.};    // Reference lengths
    constexpr auto l_data = std::array{2., 4., 6.};        // Current lengths

    const auto k_host = Kokkos::View<const double[3], Kokkos::HostSpace>(k_data.data());
    const auto l_ref_host = Kokkos::View<const double[3], Kokkos::HostSpace>(l_ref_data.data());
    const auto l_host = Kokkos::View<const double[3], Kokkos::HostSpace>(l_data.data());

    const auto k_mirror = Kokkos::create_mirror(k);
    const auto l_ref_mirror = Kokkos::create_mirror(l_ref);
    const auto l_mirror = Kokkos::create_mirror(l);

    Kokkos::deep_copy(k_mirror, k_host);
    Kokkos::deep_copy(l_ref_mirror, l_ref_host);
    Kokkos::deep_copy(l_mirror, l_host);

    Kokkos::deep_copy(k, k_mirror);
    Kokkos::deep_copy(l_ref, l_ref_mirror);
    Kokkos::deep_copy(l, l_mirror);

    Kokkos::parallel_for(
        "CalculateForceCoefficients", 3, CalculateForceCoefficients{k, l_ref, l, c1, c2}
    );

    constexpr auto c1_exact_data = std::array{
        100. * (1. / 2. - 1.),  // -50
        200. * (5. / 4. - 1.),  // 50
        300. * (3. / 6. - 1.)   // -150
    };
    constexpr auto c2_exact_data = std::array{
        100. * 1. / (2. * 2. * 2.),  // 12.5
        200. * 5. / (4. * 4. * 4.),  // 3.125
        300. * 3. / (6. * 6. * 6.)   // 1.389
    };

    const auto c1_exact = Kokkos::View<const double[3], Kokkos::HostSpace>(c1_exact_data.data());
    const auto c2_exact = Kokkos::View<const double[3], Kokkos::HostSpace>(c2_exact_data.data());

    const auto c1_result = Kokkos::create_mirror(c1);
    const auto c2_result = Kokkos::create_mirror(c2);
    Kokkos::deep_copy(c1_result, c1);
    Kokkos::deep_copy(c2_result, c2);

    CompareWithExpected(c1_result, c1_exact);
    CompareWithExpected(c2_result, c2_exact);
}

}  // namespace openturbine::tests