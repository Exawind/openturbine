#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/test_calculate.hpp"
#include "system/springs/calculate_length.hpp"

void TestCalclateLengthTests_ThreeElements() {
    const auto l = Kokkos::View<double[3]>("l");
    Kokkos::parallel_for(
        "CalculateLength", 1,
        KOKKOS_LAMBDA(const size_t) {
            constexpr auto r0_data = Kokkos::Array<double, 3>{1., 0., 0.};
            constexpr auto r1_data = Kokkos::Array<double, 3>{3., 4., 0.};
            constexpr auto r2_data = Kokkos::Array<double, 3>{1., 2., 2.};

            const auto r0 = Kokkos::View<double[3]>::const_type(r0_data.data());
            const auto r1 = Kokkos::View<double[3]>::const_type(r1_data.data());
            const auto r2 = Kokkos::View<double[3]>::const_type(r2_data.data());

            l(0) = openturbine::springs::CalculateLength(r0);
            l(1) = openturbine::springs::CalculateLength(r1);
            l(2) = openturbine::springs::CalculateLength(r2);
        }
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
