#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "solver/compute_k_row_ptrs.hpp"

namespace openturbine::tests {

TEST(ComputeKRowPtrs, OneElementOneNode) {
    constexpr auto K_num_rows = 6U;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[1]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[1]>("node_freedom_map_table");
    Kokkos::deep_copy(node_freedom_map_table, 0UL);

    const auto num_nodes_per_element = Kokkos::View<size_t[1]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 1UL);

    const auto node_state_indices = Kokkos::View<size_t[1][1]>("node_state_indices");
    Kokkos::deep_copy(node_state_indices, 0UL);

    const auto row_ptrs = ComputeKRowPtrs<Kokkos::View<size_t*>>(
        K_num_rows, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto row = 0U; row < 7U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row), row * 6U);
    }
}

TEST(ComputeKRowPtrs, OneElementTwoNodes) {
    constexpr auto K_num_rows = 12U;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[2]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[2]>("node_freedom_map_table");
    constexpr auto node_freedom_map_table_host_data = std::array{0UL, 6UL};
    const auto node_freedom_map_table_host = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        node_freedom_map_table_host_data.data()
    );
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    const auto num_nodes_per_element = Kokkos::View<size_t[1]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 2UL);

    const auto node_state_indices = Kokkos::View<size_t[1][2]>("node_state_indices");
    constexpr auto node_state_indices_host_data = std::array{0UL, 1UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[1][2], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    const auto row_ptrs = ComputeKRowPtrs<Kokkos::View<size_t*>>(
        K_num_rows, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto row = 0U; row < 13U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row), row * 12U);
    }
}

TEST(ComputeKRowPtrs, TwoElementTwoNodesNoOverlap) {
    constexpr auto K_num_rows = 24U;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[4]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[4]>("node_freedom_map_table");
    constexpr auto node_freedom_map_table_host_data = std::array{0UL, 6UL, 12UL, 18UL};
    const auto node_freedom_map_table_host = Kokkos::View<size_t[4], Kokkos::HostSpace>::const_type(
        node_freedom_map_table_host_data.data()
    );
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    const auto num_nodes_per_element = Kokkos::View<size_t[2]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 2UL);

    const auto node_state_indices = Kokkos::View<size_t[2][2]>("node_state_indices");
    constexpr auto node_state_indices_host_data = std::array{0UL, 1UL, 2UL, 3UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[2][2], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    const auto row_ptrs = ComputeKRowPtrs<Kokkos::View<size_t*>>(
        K_num_rows, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto row = 0U; row < 25U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row), row * 12U);
    }
}

TEST(ComputeKRowPtrs, TwoElementTwoNodesOverlap) {
    constexpr auto K_num_rows = 18U;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[3]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[3]>("node_freedom_map_table");
    constexpr auto node_freedom_map_table_host_data = std::array{0UL, 6UL, 12UL};
    const auto node_freedom_map_table_host = Kokkos::View<size_t[3], Kokkos::HostSpace>::const_type(
        node_freedom_map_table_host_data.data()
    );
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    const auto num_nodes_per_element = Kokkos::View<size_t[2]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 2UL);

    const auto node_state_indices = Kokkos::View<size_t[2][2]>("node_state_indices");
    constexpr auto node_state_indices_host_data = std::array{0UL, 1UL, 1UL, 2UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[2][2], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    const auto row_ptrs = ComputeKRowPtrs<Kokkos::View<size_t*>>(
        K_num_rows, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto row = 0U; row < 7U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row), row * 12U);
    }

    for (auto row = 0U; row < 6U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row + 7U), 72U + (row + 1) * 18U);
    }

    for (auto row = 0U; row < 6U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row + 13U), 180U + (row + 1) * 12U);
    }
}
/*
TEST(PopulateSparseRowPtrs, TwoElements) {
    auto num_nodes = Kokkos::View<size_t[2]>("num_nodes");
    auto num_nodes_host = Kokkos::create_mirror(num_nodes);
    num_nodes_host(0) = size_t{5U};
    num_nodes_host(1) = size_t{3U};
    Kokkos::deep_copy(num_nodes, num_nodes_host);

    constexpr auto elem1_rows = 5 * 6;
    constexpr auto elem2_rows = 3 * 6;
    constexpr auto num_rows = elem1_rows + elem2_rows;
    auto row_ptrs = Kokkos::View<int[num_rows + 1]>("row_ptrs");

    Kokkos::parallel_for(1, PopulateSparseRowPtrs<decltype(row_ptrs)>{num_nodes, row_ptrs});

    auto row_ptrs_host = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_host, row_ptrs);
    for (int row = 0; row < elem1_rows + 1; ++row) {
        EXPECT_EQ(row_ptrs_host(row), elem1_rows * row);
    }
    for (int row = elem1_rows * (elem1_rows + 1); row < elem1_rows + 1 + elem2_rows; ++row) {
        EXPECT_EQ(
            row_ptrs_host(row),
            elem1_rows * (elem1_rows + 1) + elem2_rows * (row - elem1_rows * (elem1_rows + 1))
        );
    }
}
*/
}  // namespace openturbine::tests
