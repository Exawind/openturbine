#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/compute_k_col_inds.hpp"

namespace openturbine::tests {

TEST(ComputeKColInds, OneElementOneNode) {
    constexpr auto K_num_non_zero = 36UL;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[1]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[1]>("node_freedom_map_table");
    Kokkos::deep_copy(node_freedom_map_table, 0UL);

    const auto num_nodes_per_element = Kokkos::View<size_t[1]>("num_nodes_per_element");
    Kokkos::deep_copy(num_nodes_per_element, 1UL);

    const auto node_state_indices = Kokkos::View<size_t[1][1]>("node_state_indices");
    Kokkos::deep_copy(node_state_indices, 0UL);

    const auto K_row_ptrs = Kokkos::View<int[7]>("K_row_ptrs");
    constexpr auto K_row_ptrs_host_data = std::array{0, 6, 12, 18, 24, 30, 36};
    const auto K_row_ptrs_host =
        Kokkos::View<int[7], Kokkos::HostSpace>::const_type(K_row_ptrs_host_data.data());
    const auto K_row_ptrs_mirror = Kokkos::create_mirror(K_row_ptrs);
    Kokkos::deep_copy(K_row_ptrs_mirror, K_row_ptrs_host);
    Kokkos::deep_copy(K_row_ptrs, K_row_ptrs_mirror);

    const auto col_inds = ComputeKColInds<Kokkos::View<int[7]>, Kokkos::View<int*>>(
        K_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, K_row_ptrs
    );

    const auto col_inds_mirror = Kokkos::create_mirror(col_inds);
    Kokkos::deep_copy(col_inds_mirror, col_inds);

    for (auto row = 0U; row < 6U; ++row) {
        for (auto col = 0U; col < 6U; ++col) {
            EXPECT_EQ(col_inds_mirror(row * 6U + col), col);
        }
    }
}

TEST(ComputeKColInds, OneElementTwoNodes) {
    constexpr auto K_num_non_zero = 144UL;

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

    const auto K_row_ptrs = Kokkos::View<int[13]>("K_row_ptrs");
    constexpr auto K_row_ptrs_host_data =
        std::array{0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144};
    const auto K_row_ptrs_host =
        Kokkos::View<int[13], Kokkos::HostSpace>::const_type(K_row_ptrs_host_data.data());
    const auto K_row_ptrs_mirror = Kokkos::create_mirror(K_row_ptrs);
    Kokkos::deep_copy(K_row_ptrs_mirror, K_row_ptrs_host);
    Kokkos::deep_copy(K_row_ptrs, K_row_ptrs_mirror);

    const auto K_values = Kokkos::View<double*>("K_values", K_num_non_zero);

    const auto col_inds = ComputeKColInds<Kokkos::View<int[13]>, Kokkos::View<int*>>(
        K_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, K_row_ptrs
    );

    KokkosSparse::sort_crs_matrix(K_row_ptrs, col_inds, K_values);

    const auto col_inds_mirror = Kokkos::create_mirror(col_inds);
    Kokkos::deep_copy(col_inds_mirror, col_inds);

    for (auto row = 0U; row < 12U; ++row) {
        for (auto col = 0U; col < 12U; ++col) {
            EXPECT_EQ(col_inds_mirror(row * 12U + col), col);
        }
    }
}

TEST(ComputeKColInds, TwoElementsTwoNodesNoOverlap) {
    constexpr auto K_num_non_zero = 288UL;

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

    const auto K_row_ptrs = Kokkos::View<int[25]>("K_row_ptrs");
    constexpr auto K_row_ptrs_host_data =
        std::array{0,   12,  24,  36,  48,  60,  72,  84,  96,  108, 120, 132, 144,
                   156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288};
    const auto K_row_ptrs_host =
        Kokkos::View<int[25], Kokkos::HostSpace>::const_type(K_row_ptrs_host_data.data());
    const auto K_row_ptrs_mirror = Kokkos::create_mirror(K_row_ptrs);
    Kokkos::deep_copy(K_row_ptrs_mirror, K_row_ptrs_host);
    Kokkos::deep_copy(K_row_ptrs, K_row_ptrs_mirror);

    const auto K_values = Kokkos::View<double*>("K_values", K_num_non_zero);

    const auto col_inds = ComputeKColInds<Kokkos::View<int[25]>, Kokkos::View<int*>>(
        K_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, K_row_ptrs
    );

    KokkosSparse::sort_crs_matrix(K_row_ptrs, col_inds, K_values);

    const auto col_inds_mirror = Kokkos::create_mirror(col_inds);
    Kokkos::deep_copy(col_inds_mirror, col_inds);

    for (auto row = 0U; row < 12U; ++row) {
        for (auto col = 0U; col < 12U; ++col) {
            EXPECT_EQ(col_inds_mirror(row * 12U + col), col);
        }
    }

    for (auto row = 0U; row < 12U; ++row) {
        for (auto col = 0U; col < 12U; ++col) {
            EXPECT_EQ(col_inds_mirror(144U + row * 12U + col), 12U + col);
        }
    }
}

TEST(ComputeKColInds, TwoElementsTwoNodesOverlap) {
    constexpr auto K_num_non_zero = 252UL;

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

    const auto K_row_ptrs = Kokkos::View<int[19]>("K_row_ptrs");
    constexpr auto K_row_ptrs_host_data =
        std::array{0,   12,  24,  36,  48,  60,  72,  90,  108, 126,
                   144, 162, 180, 192, 204, 216, 228, 240, 252};
    const auto K_row_ptrs_host =
        Kokkos::View<int[19], Kokkos::HostSpace>::const_type(K_row_ptrs_host_data.data());
    const auto K_row_ptrs_mirror = Kokkos::create_mirror(K_row_ptrs);
    Kokkos::deep_copy(K_row_ptrs_mirror, K_row_ptrs_host);
    Kokkos::deep_copy(K_row_ptrs, K_row_ptrs_mirror);

    const auto K_values = Kokkos::View<double*>("K_values", K_num_non_zero);

    const auto col_inds = ComputeKColInds<Kokkos::View<int[19]>, Kokkos::View<int*>>(
        K_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, num_nodes_per_element,
        node_state_indices, K_row_ptrs
    );

    KokkosSparse::sort_crs_matrix(K_row_ptrs, col_inds, K_values);

    const auto col_inds_mirror = Kokkos::create_mirror(col_inds);
    Kokkos::deep_copy(col_inds_mirror, col_inds);

    for (auto row = 0U; row < 6U; ++row) {
        for (auto col = 0U; col < 12U; ++col) {
            EXPECT_EQ(col_inds_mirror(row * 12U + col), col);
        }
    }

    for (auto row = 0U; row < 6U; ++row) {
        for (auto col = 0U; col < 18U; ++col) {
            EXPECT_EQ(col_inds_mirror(72U + row * 18U + col), col);
        }
    }

    for (auto row = 0U; row < 6U; ++row) {
        for (auto col = 0U; col < 12U; ++col) {
            EXPECT_EQ(col_inds_mirror(180U + row * 12U + col), 6U + col);
        }
    }
}
}  // namespace openturbine::tests
