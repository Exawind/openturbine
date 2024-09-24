#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/state/calculate_displacement.hpp"

namespace openturbine::tests {
TEST(CalculateDisplacement, OneNode) {
    constexpr auto h = 2.;
    constexpr auto q_delta_host_data = std::array{1., 2., 3., 4., 5., 6.};
    const auto q_delta_host =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(q_delta_host_data.data());
    const auto q_delta = Kokkos::View<double[1][6]>("q_delta");
    const auto q_delta_mirror = Kokkos::create_mirror(q_delta);
    Kokkos::deep_copy(q_delta_mirror, q_delta_host);
    Kokkos::deep_copy(q_delta, q_delta_mirror);

    constexpr auto q_prev_host_data = std::array{7., 8., 9., 10., 11., 12., 13.};
    const auto q_prev_host =
        Kokkos::View<double[1][7], Kokkos::HostSpace>::const_type(q_prev_host_data.data());
    const auto q_prev = Kokkos::View<double[1][7]>("q_prev");
    const auto q_prev_mirror = Kokkos::create_mirror(q_prev);
    Kokkos::deep_copy(q_prev_mirror, q_prev_host);
    Kokkos::deep_copy(q_prev, q_prev_mirror);

    const auto q = Kokkos::View<double[1][7]>("q");

    Kokkos::parallel_for("CalculateDisplacement", 1, CalculateDisplacement{h, q_delta, q_prev, q});

    constexpr auto q_exact_data = std::array{9.,
                                             12.,
                                             15.,
                                             -20.510952969318009,
                                             -6.4827969278626361,
                                             -5.141528734877415,
                                             -6.696180594260384};
    const auto q_exact =
        Kokkos::View<double[1][7], Kokkos::HostSpace>::const_type(q_exact_data.data());
    const auto q_mirror = Kokkos::create_mirror(q);
    Kokkos::deep_copy(q_mirror, q);
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(q_mirror(0, i), q_exact(0, i), 1.e-14);
    }
}
}  // namespace openturbine::tests
