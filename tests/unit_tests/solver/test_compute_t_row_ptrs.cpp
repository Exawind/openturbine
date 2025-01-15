#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/compute_t_row_ptrs.hpp"

namespace openturbine::tests {

TEST(ComputeTRowPtrs, OneNode) {
    constexpr auto T_num_rows = 6U;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[1]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[1]>("node_freedom_map_table");
    Kokkos::deep_copy(node_freedom_map_table, 0UL);

    const auto row_ptrs = ComputeTRowPtrs<Kokkos::View<size_t*>>(
        T_num_rows, node_freedom_allocation_table, node_freedom_map_table
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(row_ptrs_mirror(i), i * 6U);
    }
}

TEST(ComputeTRowPtrs, TwoNodes) {
    constexpr auto T_num_rows = 12U;

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

    const auto row_ptrs = ComputeTRowPtrs<Kokkos::View<size_t*>>(
        T_num_rows, node_freedom_allocation_table, node_freedom_map_table
    );

    const auto row_ptrs_mirror = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_mirror, row_ptrs);

    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_EQ(row_ptrs_mirror(i), i * 6U);
    }

    for (auto i = 0U; i < 6U; ++i) {
        EXPECT_EQ(row_ptrs_mirror(i + 7U), 36U + (i + 1) * 6U);
    }
}

/*
TEST(PopulateTangentRowPtrs, TwoNodeSystem) {
    constexpr auto num_system_nodes = 2U;
    constexpr auto num_system_dofs = 6U * num_system_nodes;
    constexpr auto num_entries = num_system_dofs + 1U;
    const auto row_ptrs = Kokkos::View<size_t[num_entries]>("row_ptrs");
    Kokkos::parallel_for(
        "PopulateTangentRowPtrs", 1, PopulateTangentRowPtrs<size_t>{num_system_nodes, row_ptrs}
    );

    constexpr auto exact_row_ptrs_data =
        std::array{0U, 6U, 12U, 18U, 24U, 30U, 36U, 42U, 48U, 54U, 60U, 66U, 72U, 78U};
    const auto exact_row_ptrs =
        Kokkos::View<const unsigned[num_entries], Kokkos::HostSpace>(exact_row_ptrs_data.data());
    const auto row_ptrs_host = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_host, row_ptrs);
    for (auto i = 0U; i < num_entries; ++i) {
        EXPECT_EQ(row_ptrs_host(i), exact_row_ptrs(i));
    }
}
*/
}  // namespace openturbine::tests
