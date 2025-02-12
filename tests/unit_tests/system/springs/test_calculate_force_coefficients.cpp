#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/test_calculate.hpp"
#include "system/springs/calculate_force_coefficients.hpp"

namespace {

void TestCalculateForceCoefficientsTests_ThreeElements() {
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
        "CalculateForceCoefficients", 3,
        KOKKOS_LAMBDA(const size_t i_elem) {
            c1(i_elem) = openturbine::springs::CalculateForceCoefficient1(k(i_elem), l_ref(i_elem), l(i_elem));
            c2(i_elem) = openturbine::springs::CalculateForceCoefficient2(k(i_elem), l_ref(i_elem), l(i_elem));
        }
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

    openturbine::tests::CompareWithExpected(c1_result, c1_exact);
    openturbine::tests::CompareWithExpected(c2_result, c2_exact);
}

}  // namespace

namespace openturbine::tests {

TEST(CalculateForceCoefficientsTests, ThreeElements) {
    TestCalculateForceCoefficientsTests_ThreeElements();
}

}  // namespace openturbine::tests
