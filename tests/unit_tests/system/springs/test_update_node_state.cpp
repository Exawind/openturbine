#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/springs/update_node_state.hpp"

namespace openturbine::tests {

TEST(UpdateNodeStateSpringsTests, TwoNodes) {
    const auto indices = Kokkos::View<size_t[1][2]>("node_state_indices");
    constexpr auto indices_data = std::array<size_t, 2>{0U, 1U};
    const auto indices_host =
        Kokkos::View<const size_t[1][2], Kokkos::HostSpace>(indices_data.data());
    const auto indices_mirror = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_mirror, indices_host);
    Kokkos::deep_copy(indices, indices_mirror);

    const auto Q = Kokkos::View<double[2][7]>("Q");
    constexpr auto Q_data = std::array{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.};
    const auto Q_host = Kokkos::View<const double[2][7], Kokkos::HostSpace>(Q_data.data());
    const auto Q_mirror = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Q_mirror, Q_host);
    Kokkos::deep_copy(Q, Q_mirror);

    const auto u1 = Kokkos::View<double[1][3]>("u1");
    const auto u2 = Kokkos::View<double[1][3]>("u2");

    Kokkos::parallel_for("UpdateNodeState", 1, springs::UpdateNodeState{indices, u1, u2, Q});

    const auto u1_mirror = Kokkos::create_mirror(u1);
    const auto u2_mirror = Kokkos::create_mirror(u2);
    Kokkos::deep_copy(u1_mirror, u1);
    Kokkos::deep_copy(u2_mirror, u2);
    for (auto i = 0U; i < 3; ++i) {
        EXPECT_EQ(u1_mirror(0, i), Q_host(0, i));
    }
    for (auto i = 0U; i < 3; ++i) {
        EXPECT_EQ(u2_mirror(0, i), Q_host(1, i));
    }
}

}  // namespace openturbine::tests
