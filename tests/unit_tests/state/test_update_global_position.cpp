
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "state/update_global_position.hpp"

namespace openturbine::tests {
TEST(UpdateGlobalPosition, OneNode) {
    const auto q = Kokkos::View<double[1][7]>("q");
    constexpr auto q_host_data = std::array{1., 2., 3., 4., 5., 6., 7.};
    const auto q_host =
        Kokkos::View<double[1][7], Kokkos::HostSpace>::const_type(q_host_data.data());
    const auto q_mirror = Kokkos::create_mirror(q);
    Kokkos::deep_copy(q_mirror, q_host);
    Kokkos::deep_copy(q, q_mirror);

    const auto x0 = Kokkos::View<double[1][7]>("x0");
    constexpr auto x0_host_data = std::array{8., 9., 10., 11., 12., 13., 14.};
    const auto x0_host =
        Kokkos::View<double[1][7], Kokkos::HostSpace>::const_type(x0_host_data.data());
    const auto x0_mirror = Kokkos::create_mirror(x0);
    Kokkos::deep_copy(x0_mirror, x0_host);
    Kokkos::deep_copy(x0, x0_mirror);

    const auto x = Kokkos::View<double[1][7]>("x");

    Kokkos::parallel_for("UpdateGlobalPosition", 1, UpdateGlobalPosition{q, x0, x});

    constexpr auto x_exact_data = std::array{9., 11., 13., -192., 96., 132., 126.};
    const auto x_exact =
        Kokkos::View<double[1][7], Kokkos::HostSpace>::const_type(x_exact_data.data());
    const auto x_mirror = Kokkos::create_mirror(x);
    Kokkos::deep_copy(x_mirror, x);
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(x_mirror(0, i), x_exact(0, i), 1.e-14);
    }
}

}  // namespace openturbine::tests
