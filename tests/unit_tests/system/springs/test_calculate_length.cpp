#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/test_calculate.hpp"
#include "system/springs/calculate_length.hpp"

void TestCalclateLengthTests_ThreeElements() {
    const auto r = Kokkos::View<double[3][3]>("r");
    constexpr auto r_data = std::array{
        1., 0., 0.,  // Element 1: length = 1.
        3., 4., 0.,  // Element 2: length = 5.
        1., 2., 2.   // Element 3: length = 3.
    };
    const auto r_host = Kokkos::View<const double[3][3], Kokkos::HostSpace>(r_data.data());
    const auto r_mirror = Kokkos::create_mirror(r);
    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(r, r_mirror);

    const auto l = Kokkos::View<double[3]>("l");
    Kokkos::parallel_for(
        "CalculateLength", 3,
        KOKKOS_LAMBDA(const size_t i_elem) { openturbine::springs::CalculateLength{i_elem, r, l}(); }
    );

    constexpr auto l_exact_data = std::array{1., 5., 3.};
    const auto l_exact = Kokkos::View<const double[3], Kokkos::HostSpace>(l_exact_data.data());

    const auto l_mirror = Kokkos::create_mirror(l);
    Kokkos::deep_copy(l_mirror, l);

    openturbine::tests::CompareWithExpected(l_mirror, l_exact);
}

namespace openturbine::tests {

TEST(CalculateLengthTests, ThreeElements) {
    TestCalclateLengthTests_ThreeElements();
}

}  // namespace openturbine::tests
