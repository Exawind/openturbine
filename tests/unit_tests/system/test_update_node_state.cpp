#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/system/update_node_state.hpp"

namespace openturbine::tests {

inline void CompareInOrder(
    const Kokkos::View<const double***>::host_mirror_type& result,
    const Kokkos::View<const double**, Kokkos::HostSpace>& expected
) {
    for (auto j = 0U; j < result.extent(1); ++j) {
        EXPECT_EQ(result(0, 0, j), expected(0, j));
    }
    for (auto j = 0U; j < result.extent(1); ++j) {
        EXPECT_EQ(result(0, 1, j), expected(1, j));
    }
}

TEST(UpdateNodeStateTests, TwoNodes_InOrder) {
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

    const auto V = Kokkos::View<double[2][6]>("V");
    constexpr auto V_data = std::array{15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.};
    const auto V_host = Kokkos::View<const double[2][6], Kokkos::HostSpace>(V_data.data());
    const auto V_mirror = Kokkos::create_mirror(V);
    Kokkos::deep_copy(V_mirror, V_host);
    Kokkos::deep_copy(V, V_mirror);

    const auto A = Kokkos::View<double[2][6]>("A");
    constexpr auto A_data = std::array{27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.};
    const auto A_host = Kokkos::View<const double[2][6], Kokkos::HostSpace>(A_data.data());
    const auto A_mirror = Kokkos::create_mirror(A);
    Kokkos::deep_copy(A_mirror, A_host);
    Kokkos::deep_copy(A, A_mirror);

    const auto node_u = Kokkos::View<double[1][2][7]>("node_u");
    const auto node_u_dot = Kokkos::View<double[1][2][6]>("node_u_dot");
    const auto node_u_ddot = Kokkos::View<double[1][2][6]>("node_u_ddot");

    Kokkos::parallel_for(
        "UpdateNodeStateBeamElement", 2,
        UpdateNodeStateBeamElement{0U, indices, node_u, node_u_dot, node_u_ddot, Q, V, A}
    );

    const auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u);
    CompareInOrder(node_u_mirror, Q_host);

    const auto node_u_dot_mirror = Kokkos::create_mirror(node_u_dot);
    Kokkos::deep_copy(node_u_dot_mirror, node_u_dot);
    CompareInOrder(node_u_dot_mirror, V_host);

    const auto node_u_ddot_mirror = Kokkos::create_mirror(node_u_ddot);
    Kokkos::deep_copy(node_u_ddot_mirror, node_u_ddot);
    CompareInOrder(node_u_ddot_mirror, A_host);
}

inline void CompareOutOfOrder(
    const Kokkos::View<const double***>::host_mirror_type& result,
    const Kokkos::View<const double**, Kokkos::HostSpace>& expected
) {
    for (auto j = 0U; j < result.extent(1); ++j) {
        EXPECT_EQ(result(0, 0, j), expected(1, j));
    }
    for (auto j = 0U; j < result.extent(1); ++j) {
        EXPECT_EQ(result(0, 1, j), expected(0, j));
    }
}

TEST(UpdateNodeStateTests, TwoNodes_OutOfOrder) {
    const auto indices = Kokkos::View<size_t[1][2]>("node_state_indices");
    constexpr auto indices_data = std::array<size_t, 2>{1U, 0U};
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

    const auto V = Kokkos::View<double[2][6]>("V");
    constexpr auto V_data = std::array{15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.};
    const auto V_host = Kokkos::View<const double[2][6], Kokkos::HostSpace>(V_data.data());
    const auto V_mirror = Kokkos::create_mirror(V);
    Kokkos::deep_copy(V_mirror, V_host);
    Kokkos::deep_copy(V, V_mirror);

    const auto A = Kokkos::View<double[2][6]>("A");
    constexpr auto A_data = std::array{27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.};
    const auto A_host = Kokkos::View<const double[2][6], Kokkos::HostSpace>(A_data.data());
    const auto A_mirror = Kokkos::create_mirror(A);
    Kokkos::deep_copy(A_mirror, A_host);
    Kokkos::deep_copy(A, A_mirror);

    const auto node_u = Kokkos::View<double[1][2][7]>("node_u");
    const auto node_u_dot = Kokkos::View<double[1][2][6]>("node_u_dot");
    const auto node_u_ddot = Kokkos::View<double[1][2][6]>("node_u_ddot");

    Kokkos::parallel_for(
        "UpdateNodeStateBeamElement", 2,
        UpdateNodeStateBeamElement{0U, indices, node_u, node_u_dot, node_u_ddot, Q, V, A}
    );

    const auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u);
    CompareOutOfOrder(node_u_mirror, Q_host);

    const auto node_u_dot_mirror = Kokkos::create_mirror(node_u_dot);
    Kokkos::deep_copy(node_u_dot_mirror, node_u_dot);
    CompareOutOfOrder(node_u_dot_mirror, V_host);

    const auto node_u_ddot_mirror = Kokkos::create_mirror(node_u_ddot);
    Kokkos::deep_copy(node_u_ddot_mirror, node_u_ddot);
    CompareOutOfOrder(node_u_ddot_mirror, A_host);
}

TEST(UpdateNodeStateTests, SingleMassElement) {
    const auto indices = Kokkos::View<size_t[1]>("state_indices");
    constexpr auto indices_data = std::array<size_t, 1>{0U};
    const auto indices_host = Kokkos::View<const size_t[1], Kokkos::HostSpace>(indices_data.data());
    const auto indices_mirror = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_mirror, indices_host);
    Kokkos::deep_copy(indices, indices_mirror);

    const auto Q = Kokkos::View<double[1][7]>("Q");
    constexpr auto Q_data = std::array{1., 2., 3., 4., 5., 6., 7.};
    const auto Q_host = Kokkos::View<const double[1][7], Kokkos::HostSpace>(Q_data.data());
    const auto Q_mirror = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Q_mirror, Q_host);
    Kokkos::deep_copy(Q, Q_mirror);

    const auto V = Kokkos::View<double[1][6]>("V");
    constexpr auto V_data = std::array{8., 9., 10., 11., 12., 13.};
    const auto V_host = Kokkos::View<const double[1][6], Kokkos::HostSpace>(V_data.data());
    const auto V_mirror = Kokkos::create_mirror(V);
    Kokkos::deep_copy(V_mirror, V_host);
    Kokkos::deep_copy(V, V_mirror);

    const auto A = Kokkos::View<double[1][6]>("A");
    constexpr auto A_data = std::array{14., 15., 16., 17., 18., 19.};
    const auto A_host = Kokkos::View<const double[1][6], Kokkos::HostSpace>(A_data.data());
    const auto A_mirror = Kokkos::create_mirror(A);
    Kokkos::deep_copy(A_mirror, A_host);
    Kokkos::deep_copy(A, A_mirror);

    const auto u = Kokkos::View<double[1][7]>("u");
    const auto u_dot = Kokkos::View<double[1][6]>("u_dot");
    const auto u_ddot = Kokkos::View<double[1][6]>("u_ddot");

    Kokkos::parallel_for(
        "UpdateNodeStateMassElement", 1,
        UpdateNodeStateMassElement{0U, indices, u, u_dot, u_ddot, Q, V, A}
    );

    const auto u_mirror = Kokkos::create_mirror(u);
    Kokkos::deep_copy(u_mirror, u);
    for (auto j = 0U; j < 7U; ++j) {
        EXPECT_EQ(u_mirror(0, j), Q_host(0, j));
    }

    const auto u_dot_mirror = Kokkos::create_mirror(u_dot);
    Kokkos::deep_copy(u_dot_mirror, u_dot);
    for (auto j = 0U; j < 6U; ++j) {
        EXPECT_EQ(u_dot_mirror(0, j), V_host(0, j));
    }

    const auto u_ddot_mirror = Kokkos::create_mirror(u_ddot);
    Kokkos::deep_copy(u_ddot_mirror, u_ddot);
    for (auto j = 0U; j < 6U; ++j) {
        EXPECT_EQ(u_ddot_mirror(0, j), A_host(0, j));
    }
}

}  // namespace openturbine::tests
