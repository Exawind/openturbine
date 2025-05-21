#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/beams/test_calculate.hpp"
#include "system/springs/calculate_distance_components.hpp"

namespace {

void TestCalculateDistanceComponentsTests_OneElement() {
    const auto x0 = openturbine::tests::CreateView<double[3]>("x0", std::array{1., 2., 3.});
    const auto u1 = openturbine::tests::CreateView<double[3]>("u1", std::array{0.1, 0.2, 0.3});
    const auto u2 = openturbine::tests::CreateView<double[3]>("u2", std::array{0.4, 0.5, 0.6});
    const auto r = Kokkos::View<double[3]>("r");

    Kokkos::parallel_for(
        "CalculateDistanceComponents", 1,
        KOKKOS_LAMBDA(const size_t) {
            openturbine::springs::CalculateDistanceComponents<Kokkos::DefaultExecutionSpace>(
                x0, u1, u2, r
            );
        }
    );

    constexpr auto r_exact_data = std::array{1.3, 2.3, 3.3};
    const auto r_exact = Kokkos::View<double[3], Kokkos::HostSpace>::const_type(r_exact_data.data());

    const auto r_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
    openturbine::tests::CompareWithExpected(r_mirror, r_exact);
}

}  // namespace

namespace openturbine::tests {

TEST(CalculateDistanceComponentsTests, OneElement) {
    TestCalculateDistanceComponentsTests_OneElement();
}

}  // namespace openturbine::tests
