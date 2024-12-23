#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/compute_k_num_non_zero.hpp"

namespace openturbine::tests {

TEST(ComputeKNumNonZero, OneElementFiveNodes) {
    auto num_nodes_per_element = Kokkos::View<size_t[1]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 5UL);

    auto node_state_indices = Kokkos::View<size_t[1][5]>("node_state_indices");
    constexpr auto node_state_indices_host_data = std::array{0UL, 1UL, 2UL, 3UL, 4UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[1][5], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[5]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto num_non_zero =
        ComputeKNumNonZero(num_nodes_per_element, node_state_indices, node_freedom_allocation_table);

    EXPECT_EQ(num_non_zero, 900UL);
}

TEST(ComputeKNumNonZero, TwoElementsFiveNodesNoOverlap) {
    auto num_nodes_per_element = Kokkos::View<size_t[2]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 5UL);

    auto node_state_indices = Kokkos::View<size_t[2][5]>("node_state_indices");
    constexpr auto node_state_indices_host_data =
        std::array{0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[2][5], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[10]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto num_non_zero =
        ComputeKNumNonZero(num_nodes_per_element, node_state_indices, node_freedom_allocation_table);

    EXPECT_EQ(num_non_zero, 1800UL);
}

TEST(ComputeKNumNonZero, TwoElementsFiveNodesOverlap) {
    auto num_nodes_per_element = Kokkos::View<size_t[2]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 5UL);

    auto node_state_indices = Kokkos::View<size_t[2][5]>("node_state_indices");
    constexpr auto node_state_indices_host_data =
        std::array{0UL, 1UL, 2UL, 3UL, 4UL, 4UL, 5UL, 6UL, 7UL, 8UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[2][5], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[9]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto num_non_zero =
        ComputeKNumNonZero(num_nodes_per_element, node_state_indices, node_freedom_allocation_table);

    EXPECT_EQ(num_non_zero, 1764UL);
}
}  // namespace openturbine::tests
