#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "model/copy_nodes_to_state.hpp"

namespace openturbine::tests {
TEST(CopyNodesToState, OneNode_ID) {
    auto state = State(1U);
    constexpr auto x0_exact = std::array<double, 7>{};
    constexpr auto q_exact = std::array<double, 7>{};
    constexpr auto v_exact = std::array<double, 6>{};
    constexpr auto vd_exact = std::array<double, 6>{};
    auto nodes = std::vector{Node(size_t{1234U}, x0_exact, q_exact, v_exact, vd_exact)};

    CopyNodesToState(state, nodes);

    const auto id = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.ID);
    EXPECT_EQ(id(0), 1234U);
}

TEST(CopyNodesToState, OneNode_Position) {
    auto state = State(1U);
    constexpr auto x0_exact = std::array{1., 2., 3., 4., 5., 6., 7.};
    constexpr auto q_exact = std::array{8., 9., 10., 11., 12., 13., 14.};
    constexpr auto v_exact = std::array<double, 6>{};
    constexpr auto vd_exact = std::array<double, 6>{};
    auto nodes = std::vector{Node(size_t{1234U}, x0_exact, q_exact, v_exact, vd_exact)};

    CopyNodesToState(state, nodes);

    const auto x0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x0);
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);
    const auto q_prev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q_prev);
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x);

    const auto x_exact = std::array{9., 11., 13., -192., 110., 104., 140.};

    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(x0(0, i), x0_exact[i]);
        EXPECT_EQ(q(0, i), q_exact[i]);
        EXPECT_EQ(q_prev(0, i), q(0, i));
        EXPECT_EQ(x(0, i), x_exact[i]);
    }
}

TEST(CopyNodesToState, OneNode_Velocity) {
    auto state = State(1U);
    constexpr auto x0_exact = std::array<double, 7>{};
    constexpr auto q_exact = std::array<double, 7>{};
    constexpr auto v_exact = std::array{15., 16., 17., 18., 19., 20.};
    constexpr auto vd_exact = std::array{21., 22., 23., 24., 25., 26.};
    auto nodes = std::vector{Node(size_t{1234U}, x0_exact, q_exact, v_exact, vd_exact)};

    CopyNodesToState(state, nodes);

    const auto v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.v);
    const auto vd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.vd);

    for (auto i = 0U; i < 6U; ++i) {
        EXPECT_EQ(v(0, i), v_exact[i]);
        EXPECT_EQ(vd(0, i), vd_exact[i]);
    }
}
}  // namespace openturbine::tests
