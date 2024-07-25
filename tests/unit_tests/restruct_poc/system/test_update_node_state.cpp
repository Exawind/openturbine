#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/system/update_node_state.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

TEST(UpdateNodeStateTests, TwoNodes_InOrder) {
    auto indices = Kokkos::View<size_t[2]>("node_state_indices");
    auto indices_data = std::array<size_t, 2>{0u, 1u};
    auto indices_host = Kokkos::View<size_t[2], Kokkos::HostSpace>(indices_data.data());
    auto indices_mirror = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_mirror, indices_host);
    Kokkos::deep_copy(indices, indices_host);

    auto Q = Kokkos::View<double[2][7]>("Q");
    auto Q_data =
        std::array<double, 14>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.};
    auto Q_host = Kokkos::View<double[2][7], Kokkos::HostSpace>(Q_data.data());
    auto Q_mirror = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Q_mirror, Q_host);
    Kokkos::deep_copy(Q, Q_mirror);

    auto V = Kokkos::View<double[2][6]>("V");
    auto V_data = std::array<double, 12>{15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.};
    auto V_host = Kokkos::View<double[2][6], Kokkos::HostSpace>(V_data.data());
    auto V_mirror = Kokkos::create_mirror(V);
    Kokkos::deep_copy(V_mirror, V_host);
    Kokkos::deep_copy(V, V_mirror);

    auto A = Kokkos::View<double[2][6]>("A");
    auto A_data = std::array<double, 12>{27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.};
    auto A_host = Kokkos::View<double[2][6], Kokkos::HostSpace>(A_data.data());
    auto A_mirror = Kokkos::create_mirror(A);
    Kokkos::deep_copy(A_mirror, A_host);
    Kokkos::deep_copy(A, A_mirror);

    auto node_u = Kokkos::View<double[2][7]>("node_u");
    auto node_u_dot = Kokkos::View<double[2][6]>("node_u_dot");
    auto node_u_ddot = Kokkos::View<double[2][6]>("node_u_ddot");

    Kokkos::parallel_for(
        "UpdateNodeState", 2, UpdateNodeState{indices, node_u, node_u_dot, node_u_ddot, Q, V, A}
    );

    auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u);
    for (int j = 0; j < 7; ++j) {
        EXPECT_EQ(node_u_mirror(0, j), Q_host(0, j));
    }
    for (int j = 0; j < 7; ++j) {
        EXPECT_EQ(node_u_mirror(1, j), Q_host(1, j));
    }

    auto node_u_dot_mirror = Kokkos::create_mirror(node_u_dot);
    Kokkos::deep_copy(node_u_dot_mirror, node_u_dot);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_dot_mirror(0, j), V_host(0, j));
    }
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_dot_mirror(1, j), V_host(1, j));
    }

    auto node_u_ddot_mirror = Kokkos::create_mirror(node_u_ddot);
    Kokkos::deep_copy(node_u_ddot_mirror, node_u_ddot);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_ddot_mirror(0, j), A_host(0, j));
    }
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_ddot_mirror(1, j), A_host(1, j));
    }
}

TEST(UpdateNodeStateTests, TwoNodes_OutOfOrder) {
    auto indices = Kokkos::View<size_t[2]>("node_state_indices");
    auto indices_data = std::array<size_t, 2>{1u, 0u};
    auto indices_host = Kokkos::View<size_t[2], Kokkos::HostSpace>(indices_data.data());
    auto indices_mirror = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_mirror, indices_host);
    Kokkos::deep_copy(indices, indices_host);

    auto Q = Kokkos::View<double[2][7]>("Q");
    auto Q_data =
        std::array<double, 14>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.};
    auto Q_host = Kokkos::View<double[2][7], Kokkos::HostSpace>(Q_data.data());
    auto Q_mirror = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Q_mirror, Q_host);
    Kokkos::deep_copy(Q, Q_mirror);

    auto V = Kokkos::View<double[2][6]>("V");
    auto V_data = std::array<double, 12>{15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.};
    auto V_host = Kokkos::View<double[2][6], Kokkos::HostSpace>(V_data.data());
    auto V_mirror = Kokkos::create_mirror(V);
    Kokkos::deep_copy(V_mirror, V_host);
    Kokkos::deep_copy(V, V_mirror);

    auto A = Kokkos::View<double[2][6]>("A");
    auto A_data = std::array<double, 12>{27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.};
    auto A_host = Kokkos::View<double[2][6], Kokkos::HostSpace>(A_data.data());
    auto A_mirror = Kokkos::create_mirror(A);
    Kokkos::deep_copy(A_mirror, A_host);
    Kokkos::deep_copy(A, A_mirror);

    auto node_u = Kokkos::View<double[2][7]>("node_u");
    auto node_u_dot = Kokkos::View<double[2][6]>("node_u_dot");
    auto node_u_ddot = Kokkos::View<double[2][6]>("node_u_ddot");

    Kokkos::parallel_for(
        "UpdateNodeState", 2, UpdateNodeState{indices, node_u, node_u_dot, node_u_ddot, Q, V, A}
    );

    auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u);
    for (int j = 0; j < 7; ++j) {
        EXPECT_EQ(node_u_mirror(0, j), Q_host(1, j));
    }
    for (int j = 0; j < 7; ++j) {
        EXPECT_EQ(node_u_mirror(1, j), Q_host(0, j));
    }

    auto node_u_dot_mirror = Kokkos::create_mirror(node_u_dot);
    Kokkos::deep_copy(node_u_dot_mirror, node_u_dot);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_dot_mirror(0, j), V_host(1, j));
    }
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_dot_mirror(1, j), V_host(0, j));
    }

    auto node_u_ddot_mirror = Kokkos::create_mirror(node_u_ddot);
    Kokkos::deep_copy(node_u_ddot_mirror, node_u_ddot);
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_ddot_mirror(0, j), A_host(1, j));
    }
    for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(node_u_ddot_mirror(1, j), A_host(0, j));
    }
}

}  // namespace openturbine::restruct_poc::tests