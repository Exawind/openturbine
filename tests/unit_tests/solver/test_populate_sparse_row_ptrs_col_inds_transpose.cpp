#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "solver/populate_sparse_row_ptrs_col_inds_transpose.hpp"

namespace openturbine::tests {

TEST(PopulateSparseRowPtrsColInds_Transpose, DenseSquare) {
    constexpr auto rows = 5U;
    constexpr auto cols = 5U;
    constexpr auto nnz = rows * cols;

    constexpr auto row_ptrs_host_data = std::array<size_t, rows + 1U>{0U, 5U, 10U, 15U, 20U, 25U};
    const auto row_ptrs = Kokkos::View<size_t[rows + 1U]>("row_ptrs");
    const auto row_ptrs_host =
        Kokkos::View<const size_t[rows + 1U], Kokkos::HostSpace>(row_ptrs_host_data.data());
    Kokkos::deep_copy(row_ptrs, row_ptrs_host);

    constexpr auto col_inds_host_data =
        std::array<size_t, nnz>{0U, 1U, 2U, 3U, 4U, 0U, 1U, 2U, 3U, 4U, 0U, 1U, 2U,
                                3U, 4U, 0U, 1U, 2U, 3U, 4U, 0U, 1U, 2U, 3U, 4U};
    const auto col_inds = Kokkos::View<size_t[nnz]>("col_inds");
    const auto col_inds_host =
        Kokkos::View<const size_t[nnz], Kokkos::HostSpace>(col_inds_host_data.data());
    Kokkos::deep_copy(col_inds, col_inds_host);

    const auto col_count = Kokkos::View<size_t[cols]>("col_count");
    const auto temp_row_ptr = Kokkos::View<size_t[cols + 1U]>("temp_row_ptr");
    const auto row_ptrs_trans = Kokkos::View<size_t[cols + 1U]>("row_ptrs_trans");
    const auto col_inds_trans = Kokkos::View<size_t[nnz]>("col_inds_trans");

    Kokkos::parallel_for(
        "PopulateSparseRowPtrsColInds_Transpose", 1,
        PopulateSparseRowPtrsColInds_Transpose<Kokkos::View<size_t*>, Kokkos::View<size_t*>>{
            rows, cols, row_ptrs, col_inds, col_count, temp_row_ptr, row_ptrs_trans, col_inds_trans
        }
    );

    constexpr auto row_ptrs_trans_exact_data = row_ptrs_host_data;
    const auto row_ptrs_trans_exact =
        Kokkos::View<const size_t[cols + 1U], Kokkos::HostSpace>(row_ptrs_trans_exact_data.data());

    const auto row_ptrs_trans_host = Kokkos::create_mirror(row_ptrs_trans);
    Kokkos::deep_copy(row_ptrs_trans_host, row_ptrs_trans);
    for (auto i = 0U; i < cols + 1U; ++i) {
        EXPECT_EQ(row_ptrs_trans_host(i), row_ptrs_trans_exact(i));
    }

    constexpr auto col_inds_trans_exact_data = col_inds_host_data;
    const auto col_inds_trans_exact =
        Kokkos::View<const size_t[nnz], Kokkos::HostSpace>(col_inds_trans_exact_data.data());

    const auto col_inds_trans_host = Kokkos::create_mirror(col_inds_trans);
    Kokkos::deep_copy(col_inds_trans_host, col_inds_trans);
    for (auto i = 0U; i < nnz; ++i) {
        EXPECT_EQ(col_inds_trans_host(i), col_inds_trans_exact(i));
    }
}

TEST(PopulateSparseRowPtrsColInds_Transpose, DenseRectangle) {
    constexpr auto rows = 1U;
    constexpr auto cols = 5U;
    constexpr auto nnz = rows * cols;

    constexpr auto row_ptrs_host_data = std::array<size_t, rows + 1U>{0U, 5U};
    const auto row_ptrs = Kokkos::View<size_t[rows + 1U]>("row_ptrs");
    const auto row_ptrs_host =
        Kokkos::View<const size_t[rows + 1U], Kokkos::HostSpace>(row_ptrs_host_data.data());
    Kokkos::deep_copy(row_ptrs, row_ptrs_host);

    constexpr auto col_inds_host_data = std::array<size_t, nnz>{0U, 1U, 2U, 3U, 4U};
    const auto col_inds = Kokkos::View<size_t[nnz]>("col_inds");
    const auto col_inds_host =
        Kokkos::View<const size_t[nnz], Kokkos::HostSpace>(col_inds_host_data.data());
    Kokkos::deep_copy(col_inds, col_inds_host);

    const auto col_count = Kokkos::View<size_t[cols]>("col_count");
    const auto temp_row_ptr = Kokkos::View<size_t[cols + 1U]>("temp_row_ptr");
    const auto row_ptrs_trans = Kokkos::View<size_t[cols + 1U]>("row_ptrs_trans");
    const auto col_inds_trans = Kokkos::View<size_t[nnz]>("col_inds_trans");

    Kokkos::parallel_for(
        "PopulateSparseRowPtrsColInds_Transpose", 1,
        PopulateSparseRowPtrsColInds_Transpose<Kokkos::View<size_t*>, Kokkos::View<size_t*>>{
            rows, cols, row_ptrs, col_inds, col_count, temp_row_ptr, row_ptrs_trans, col_inds_trans
        }
    );

    constexpr auto row_ptrs_trans_exact_data = std::array<size_t, cols + 1U>{0U, 1U, 2U, 3U, 4U, 5U};
    const auto row_ptrs_trans_exact =
        Kokkos::View<const size_t[cols + 1U], Kokkos::HostSpace>(row_ptrs_trans_exact_data.data());

    const auto row_ptrs_trans_host = Kokkos::create_mirror(row_ptrs_trans);
    Kokkos::deep_copy(row_ptrs_trans_host, row_ptrs_trans);
    for (auto i = 0U; i < cols + 1U; ++i) {
        EXPECT_EQ(row_ptrs_trans_host(i), row_ptrs_trans_exact(i));
    }

    constexpr auto col_inds_trans_exact_data = std::array<size_t, nnz>{0U, 0U, 0U, 0U, 0U};
    const auto col_inds_trans_exact =
        Kokkos::View<const size_t[nnz], Kokkos::HostSpace>(col_inds_trans_exact_data.data());

    const auto col_inds_trans_host = Kokkos::create_mirror(col_inds_trans);
    Kokkos::deep_copy(col_inds_trans_host, col_inds_trans);
    for (auto i = 0U; i < nnz; ++i) {
        EXPECT_EQ(col_inds_trans_host(i), col_inds_trans_exact(i));
    }
}
}  // namespace openturbine::tests
