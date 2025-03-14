#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "solver/compute_row_ptrs.hpp"

namespace openturbine::tests {

TEST(ComputeRowPtrs, OneElementOneNode) {
    constexpr auto num_system_dofs = 6U;
    constexpr auto num_dofs = num_system_dofs;

    const auto active_dofs = Kokkos::View<size_t[1]>("active_dofs");
    Kokkos::deep_copy(active_dofs, 6UL);

    const auto node_freedom_map_table = Kokkos::View<size_t[1]>("node_freedom_map_table");
    Kokkos::deep_copy(node_freedom_map_table, 0UL);

    const auto num_nodes_per_element = Kokkos::View<size_t[1]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 1UL);

    const auto node_state_indices = Kokkos::View<size_t[1][1]>("node_state_indices");
    Kokkos::deep_copy(node_state_indices, 0UL);

    const auto base_active_dofs = Kokkos::View<size_t*>("base_active_dofs", 0);
    const auto target_active_dofs = Kokkos::View<size_t*>("target_active_dofs", 0);
    const auto base_node_freedom_table = Kokkos::View<size_t* [6]>("base_node_freedom_table", 0);
    const auto target_node_freedom_table = Kokkos::View<size_t* [6]>("target_node_freedom_table", 0);
    const auto row_range = Kokkos::View<Kokkos::pair<size_t, size_t>*>("row_range", 0);

    const auto row_ptrs = ComputeRowPtrs<Kokkos::View<size_t*>>(
        num_system_dofs, num_dofs, active_dofs, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table,
        target_node_freedom_table, row_range
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror_view(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto row = 0U; row < 7U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row), row * 6U);
    }
}

TEST(ComputeRowPtrs, OneElementTwoNodes) {
    constexpr auto num_system_dofs = 12U;
    constexpr auto num_dofs = num_system_dofs;

    const auto active_dofs = Kokkos::View<size_t[2]>("active_dofs");
    Kokkos::deep_copy(active_dofs, 6UL);

    const auto node_freedom_map_table = Kokkos::View<size_t[2]>("node_freedom_map_table");
    constexpr auto node_freedom_map_table_host_data = std::array{0UL, 6UL};
    const auto node_freedom_map_table_host = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        node_freedom_map_table_host_data.data()
    );
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror_view(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    const auto num_nodes_per_element = Kokkos::View<size_t[1]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 2UL);

    const auto node_state_indices = Kokkos::View<size_t[1][2]>("node_state_indices");
    constexpr auto node_state_indices_host_data = std::array{0UL, 1UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[1][2], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror_view(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    const auto base_active_dofs = Kokkos::View<size_t*>("base_active_dofs", 0);
    const auto target_active_dofs = Kokkos::View<size_t*>("target_active_dofs", 0);
    const auto base_node_freedom_table = Kokkos::View<size_t* [6]>("base_node_freedom_table", 0);
    const auto target_node_freedom_table = Kokkos::View<size_t* [6]>("target_node_freedom_table", 0);
    const auto row_range = Kokkos::View<Kokkos::pair<size_t, size_t>*>("row_range", 0);

    const auto row_ptrs = ComputeRowPtrs<Kokkos::View<size_t*>>(
        num_system_dofs, num_dofs, active_dofs, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table,
        target_node_freedom_table, row_range
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror_view(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto row = 0U; row < 13U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row), row * 12U);
    }
}

TEST(ComputeRowPtrs, TwoElementTwoNodesNoOverlap) {
    constexpr auto num_system_dofs = 24U;
    constexpr auto num_dofs = num_system_dofs;

    const auto active_dofs = Kokkos::View<size_t[4]>("active_dofs");
    Kokkos::deep_copy(active_dofs, 6UL);

    const auto node_freedom_map_table = Kokkos::View<size_t[4]>("node_freedom_map_table");
    constexpr auto node_freedom_map_table_host_data = std::array{0UL, 6UL, 12UL, 18UL};
    const auto node_freedom_map_table_host = Kokkos::View<size_t[4], Kokkos::HostSpace>::const_type(
        node_freedom_map_table_host_data.data()
    );
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror_view(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    const auto num_nodes_per_element = Kokkos::View<size_t[2]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 2UL);

    const auto node_state_indices = Kokkos::View<size_t[2][2]>("node_state_indices");
    constexpr auto node_state_indices_host_data = std::array{0UL, 1UL, 2UL, 3UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[2][2], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror_view(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    const auto base_active_dofs = Kokkos::View<size_t*>("base_active_dofs", 0);
    const auto target_active_dofs = Kokkos::View<size_t*>("target_active_dofs", 0);
    const auto base_node_freedom_table = Kokkos::View<size_t* [6]>("base_node_freedom_table", 0);
    const auto target_node_freedom_table = Kokkos::View<size_t* [6]>("target_node_freedom_table", 0);
    const auto row_range = Kokkos::View<Kokkos::pair<size_t, size_t>*>("row_range", 0);

    const auto row_ptrs = ComputeRowPtrs<Kokkos::View<size_t*>>(
        num_system_dofs, num_dofs, active_dofs, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table,
        target_node_freedom_table, row_range
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror_view(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto row = 0U; row < 25U; ++row) {
        EXPECT_EQ(row_ptrs_mirror(row), row * 12U);
    }
}

TEST(ComputeRowPtrs, TwoElementTwoNodesOverlap) {
    constexpr auto num_system_dofs = 18U;
    constexpr auto num_dofs = num_system_dofs;

    const auto active_dofs = Kokkos::View<size_t[3]>("active_dofs");
    Kokkos::deep_copy(active_dofs, 6UL);

    const auto node_freedom_map_table = Kokkos::View<size_t[3]>("node_freedom_map_table");
    constexpr auto node_freedom_map_table_host_data = std::array{0UL, 6UL, 12UL};
    const auto node_freedom_map_table_host = Kokkos::View<size_t[3], Kokkos::HostSpace>::const_type(
        node_freedom_map_table_host_data.data()
    );
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror_view(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    const auto num_nodes_per_element = Kokkos::View<size_t[2]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 2UL);

    const auto node_state_indices = Kokkos::View<size_t[2][2]>("node_state_indices");
    constexpr auto node_state_indices_host_data = std::array{0UL, 1UL, 1UL, 2UL};
    const auto node_state_indices_host =
        Kokkos::View<size_t[2][2], Kokkos::HostSpace>::const_type(node_state_indices_host_data.data()
        );
    const auto node_state_indices_mirror = Kokkos::create_mirror_view(node_state_indices);
    Kokkos::deep_copy(node_state_indices_mirror, node_state_indices_host);
    Kokkos::deep_copy(node_state_indices, node_state_indices_mirror);

    const auto base_active_dofs = Kokkos::View<size_t*>("base_active_dofs", 0);
    const auto target_active_dofs = Kokkos::View<size_t*>("target_active_dofs", 0);
    const auto base_node_freedom_table = Kokkos::View<size_t* [6]>("base_node_freedom_table", 0);
    const auto target_node_freedom_table = Kokkos::View<size_t* [6]>("target_node_freedom_table", 0);
    const auto row_range = Kokkos::View<Kokkos::pair<size_t, size_t>*>("row_range", 0);

    const auto row_ptrs = ComputeRowPtrs<Kokkos::View<size_t*>>(
        num_system_dofs, num_dofs, active_dofs, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, base_active_dofs, target_active_dofs, base_node_freedom_table,
        target_node_freedom_table, row_range
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror_view(row_ptrs);
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

}  // namespace openturbine::tests
