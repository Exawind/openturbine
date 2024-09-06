#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/solver/populate_sparse_row_ptrs.hpp"

namespace openturbine::tests {

TEST(PopulateSparseRowPtrs, SingleElement) {
    auto num_nodes = Kokkos::View<size_t[1]>("num_nodes");
    auto num_nodes_host = Kokkos::create_mirror(num_nodes);
    num_nodes_host(0) = size_t{5U};
    Kokkos::deep_copy(num_nodes, num_nodes_host);

    constexpr auto num_rows = 5 * 6;
    auto row_ptrs = Kokkos::View<int[num_rows + 1]>("row_ptrs");

    Kokkos::parallel_for(1, PopulateSparseRowPtrs<decltype(row_ptrs)>{num_nodes, row_ptrs});

    auto row_ptrs_host = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_host, row_ptrs);
    for (int row = 0; row < num_rows + 1; ++row) {
        EXPECT_EQ(row_ptrs_host(row), num_rows * row);
    }
}

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

}  // namespace openturbine::tests
