#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/model/copy_nodes_to_state.hpp"

namespace openturbine::tests {
TEST(CopyNodesToState, OneNode) {
    auto state = State(1U);
    constexpr auto x0_exact = std::array{1., 2., 3., 4., 5., 6., 7.};
    constexpr auto q_exact = std::array{8., 9., 10., 11., 12., 13., 14.};
    constexpr auto v_exact = std::array{15., 16., 17., 18., 19., 20.};
    constexpr auto vd_exact = std::array{21., 22., 23., 24., 25., 26.};
    auto nodes = std::vector{std::make_shared<Node>(size_t{1234U}, x0_exact, q_exact, v_exact, vd_exact)};

    CopyNodesToState(state, nodes);

    const auto id = Kokkos::create_mirror(state.ID);
    Kokkos::deep_copy(id, state.ID);
    EXPECT_EQ(id(0), 1234U);

    const auto x0 = Kokkos::create_mirror(state.x0);
    Kokkos::deep_copy(x0, state.x0);
    for(auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(x0(0, i), x0_exact[i]);
    }

    const auto q = Kokkos::create_mirror(state.q);
    Kokkos::deep_copy(q, state.q);
    for(auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(q(0, i), q_exact[i]);
    }

    const auto v = Kokkos::create_mirror(state.v);
    Kokkos::deep_copy(v, state.v);
    for(auto i = 0U; i < 6U; ++i) {
        EXPECT_EQ(v(0, i), v_exact[i]);
    }

    const auto vd = Kokkos::create_mirror(state.vd);
    Kokkos::deep_copy(vd, state.vd);
    for(auto i = 0U; i < 6U; ++i) {
        EXPECT_EQ(vd(0, i), vd_exact[i]);
    }

    const auto q_prev = Kokkos::create_mirror(state.q_prev);
    Kokkos::deep_copy(q_prev, state.q_prev);
    for(auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(q_prev(0, i), q(0, i));
    }

    const auto x = Kokkos::create_mirror(state.x);
    Kokkos::deep_copy(x, state.x);
    const auto x_exact = std::array{9., 11., 13., -192., 110., 104., 140.};
    for(auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(x(0, i), x_exact[i]);
    }
}
}
