#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/compute_t_col_inds.hpp"

namespace openturbine::tests {

TEST(ComputeTColInds, OneNode) {
    constexpr auto T_num_non_zero = 36UL;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[1]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[1]>("node_freedom_map_table");
    Kokkos::deep_copy(node_freedom_map_table, 0UL);

    const auto T_row_ptrs = Kokkos::View<size_t[7]>("T_row_ptrs");
    constexpr auto T_row_ptrs_host_data = std::array{0UL, 6UL, 12UL, 18UL, 24UL, 30UL, 36UL};
    const auto T_row_ptrs_host =
        Kokkos::View<size_t[7], Kokkos::HostSpace>::const_type(T_row_ptrs_host_data.data());
    const auto T_row_ptrs_mirror = Kokkos::create_mirror(T_row_ptrs);
    Kokkos::deep_copy(T_row_ptrs_mirror, T_row_ptrs_host);
    Kokkos::deep_copy(T_row_ptrs, T_row_ptrs_mirror);

    const auto col_inds = ComputeTColInds<Kokkos::View<size_t[7]>, Kokkos::View<int*>>(
        T_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, T_row_ptrs
    );

    const auto col_inds_mirror = Kokkos::create_mirror(col_inds);
    Kokkos::deep_copy(col_inds_mirror, col_inds);

    for (auto row = 0U; row < 6U; ++row) {
        for (auto col = 0U; col < 6U; ++col) {
            EXPECT_EQ(col_inds_mirror(row * 6U + col), col);
        }
    }
}

TEST(ComputeTColInds, TwoNodes) {
    constexpr auto T_num_non_zero = 72UL;

    const auto node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[2]>("node_freedom_allocation_table");
    Kokkos::deep_copy(node_freedom_allocation_table, FreedomSignature::AllComponents);

    const auto node_freedom_map_table = Kokkos::View<size_t[2]>("node_freedom_map_table");
    constexpr auto node_freedom_map_table_host_data = std::array{0UL, 6UL};
    const auto node_freedom_map_table_host =
        Kokkos::View<size_t[2]>::const_type(node_freedom_map_table_host_data.data());
    const auto node_freedom_map_table_mirror = Kokkos::create_mirror(node_freedom_map_table);
    Kokkos::deep_copy(node_freedom_map_table_mirror, node_freedom_map_table_host);
    Kokkos::deep_copy(node_freedom_map_table, node_freedom_map_table_mirror);

    const auto T_row_ptrs = Kokkos::View<size_t[13]>("T_row_ptrs");
    constexpr auto T_row_ptrs_host_data =
        std::array{0UL, 6UL, 12UL, 18UL, 24UL, 30UL, 36UL, 42UL, 48UL, 54UL, 60UL, 66UL, 72UL};
    const auto T_row_ptrs_host =
        Kokkos::View<size_t[13], Kokkos::HostSpace>::const_type(T_row_ptrs_host_data.data());
    const auto T_row_ptrs_mirror = Kokkos::create_mirror(T_row_ptrs);
    Kokkos::deep_copy(T_row_ptrs_mirror, T_row_ptrs_host);
    Kokkos::deep_copy(T_row_ptrs, T_row_ptrs_mirror);

    const auto col_inds = ComputeTColInds<Kokkos::View<size_t[13]>, Kokkos::View<int*>>(
        T_num_non_zero, node_freedom_allocation_table, node_freedom_map_table, T_row_ptrs
    );

    const auto col_inds_mirror = Kokkos::create_mirror(col_inds);
    Kokkos::deep_copy(col_inds_mirror, col_inds);

    for (auto row = 0U; row < 6U; ++row) {
        for (auto col = 0U; col < 6U; ++col) {
            EXPECT_EQ(col_inds_mirror(row * 6U + col), col);
        }
    }

    for (auto row = 0U; row < 6U; ++row) {
        for (auto col = 0U; col < 6U; ++col) {
            EXPECT_EQ(col_inds_mirror(36U + row * 6U + col), col + 6U);
        }
    }
}
/*
TEST(PopulateTangentIndices, TwoNodes) {
    constexpr auto num_system_nodes = 2U;
    constexpr auto num_system_dofs = num_system_nodes * 6U * 6U;
    constexpr auto node_state_indices_host_data = std::array{size_t{0U}, size_t{1U}};
    const auto node_state_indices_host =
        Kokkos::View<const size_t[num_system_nodes], Kokkos::HostSpace>(
            node_state_indices_host_data.data()
        );
    const auto node_state_indices = Kokkos::View<size_t[num_system_nodes]>("node_state_indices");
    Kokkos::deep_copy(node_state_indices, node_state_indices_host);

    const auto indices = Kokkos::View<int[num_system_dofs]>("indices");

    Kokkos::parallel_for(
        "PopulateTangentIndices", 1,
        PopulateTangentIndices{num_system_nodes, node_state_indices, indices}
    );

    constexpr auto indices_exact_data = std::array{
        0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,
        0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,  6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11,
        6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11,
    };
    const auto indices_exact =
        Kokkos::View<const int[num_system_dofs], Kokkos::HostSpace>(indices_exact_data.data());

    const auto indices_host = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_host, indices);

    for (auto i = 0U; i < num_system_dofs; ++i) {
        EXPECT_EQ(indices_host(i), indices_exact(i));
    }
}
*/
}  // namespace openturbine::tests
