#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/masses/update_node_state.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

TEST(UpdateNodeStateMassesTests, OneNode) {
    const auto indices = Kokkos::View<size_t[1]>("node_state_indices");
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
    constexpr auto V_data = std::array{15., 16., 17., 18., 19., 20.};
    const auto V_host = Kokkos::View<const double[1][6], Kokkos::HostSpace>(V_data.data());
    const auto V_mirror = Kokkos::create_mirror(V);
    Kokkos::deep_copy(V_mirror, V_host);
    Kokkos::deep_copy(V, V_mirror);

    const auto A = Kokkos::View<double[1][6]>("A");
    constexpr auto A_data = std::array{27., 28., 29., 30., 31., 32.};
    const auto A_host = Kokkos::View<const double[1][6], Kokkos::HostSpace>(A_data.data());
    const auto A_mirror = Kokkos::create_mirror(A);
    Kokkos::deep_copy(A_mirror, A_host);
    Kokkos::deep_copy(A, A_mirror);

    const auto node_u = Kokkos::View<double[1][7]>("node_u");
    const auto node_u_dot = Kokkos::View<double[1][6]>("node_u_dot");
    const auto node_u_ddot = Kokkos::View<double[1][6]>("node_u_ddot");

    Kokkos::parallel_for(
        "UpdateNodeStateElement", 1,
        masses::UpdateNodeState{indices, node_u, node_u_dot, node_u_ddot, Q, V, A}
    );

    const auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u);
    CompareWithExpected(node_u_mirror, Q_host);

    const auto node_u_dot_mirror = Kokkos::create_mirror(node_u_dot);
    Kokkos::deep_copy(node_u_dot_mirror, node_u_dot);
    CompareWithExpected(node_u_dot_mirror, V_host);

    const auto node_u_ddot_mirror = Kokkos::create_mirror(node_u_ddot);
    Kokkos::deep_copy(node_u_ddot_mirror, node_u_ddot);
    CompareWithExpected(node_u_ddot_mirror, A_host);
}
}  // namespace openturbine::tests
