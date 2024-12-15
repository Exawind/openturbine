#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/system/springs/calculate_length.hpp"

namespace openturbine::tests {

TEST(CalculateLengthTests, ThreeElements) {
    const auto r = Kokkos::View<double[3][3]>("r");
    constexpr auto r_data = std::array{
        1., 0., 0.,  // Element 1: length should be 1
        3., 4., 0.,  // Element 2: length should be 5
        1., 2., 2.   // Element 3: length should be 3
    };
    const auto r_host = Kokkos::View<const double[3][3], Kokkos::HostSpace>(r_data.data());
    const auto r_mirror = Kokkos::create_mirror(r);
    Kokkos::deep_copy(r_mirror, r_host);
    Kokkos::deep_copy(r, r_mirror);

    const auto l = Kokkos::View<double[3]>("l");
    Kokkos::parallel_for("CalculateLength", 3, CalculateLength{r, l});

    constexpr auto l_exact_data = std::array{1., 5., 3.};
    const auto l_exact = Kokkos::View<const double[3], Kokkos::HostSpace>(l_exact_data.data());

    const auto l_mirror = Kokkos::create_mirror(l);
    Kokkos::deep_copy(l_mirror, l);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(l_mirror(i), l_exact(i), 1e-10);
    }
}

}  // namespace openturbine::tests
