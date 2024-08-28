#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/copy_into_sparse_matrix.hpp"

namespace openturbine::tests {

auto createDenseMatrix_1x1() {
    auto dense = Kokkos::View<double[1][1]>("dense");
    auto dense_host_data = std::array<double, 1>{3.};
    auto dense_host = Kokkos::View<double[1][1], Kokkos::HostSpace>(dense_host_data.data());
    auto dense_mirror = Kokkos::create_mirror(dense);
    Kokkos::deep_copy(dense_mirror, dense_host);
    Kokkos::deep_copy(dense, dense_mirror);
    return dense;
}

auto createRowPtrs_1x1() {
    auto row_ptrs = Kokkos::View<int[2]>("row_ptrs");
    auto row_ptrs_host_data = std::array{0, 1};
    auto row_ptrs_host = Kokkos::View<int[2], Kokkos::HostSpace>(row_ptrs_host_data.data());
    Kokkos::deep_copy(row_ptrs, row_ptrs_host);
    return row_ptrs;
}

TEST(CopyIntoSparseMatrix, SingleEntry) {
    using device_type =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device_type, void, int>;

    constexpr auto num_rows = 1;
    constexpr auto num_columns = 1;
    constexpr auto num_non_zero = 1;

    auto dense = createDenseMatrix_1x1();

    auto values = Kokkos::View<double[num_non_zero]>("values");
    auto row_ptrs = createRowPtrs_1x1();
    auto indices = Kokkos::View<int[num_non_zero]>("indices");
    auto sparse =
        crs_matrix_type("sparse", num_rows, num_columns, num_non_zero, values, row_ptrs, indices);

    auto row_data_size = Kokkos::View<double*>::shmem_size(num_columns);
    auto col_idx_size = Kokkos::View<int*>::shmem_size(num_columns);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(num_rows, Kokkos::AUTO());
    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", sparse_matrix_policy,
        CopyIntoSparseMatrix<crs_matrix_type>{sparse, dense}
    );

    auto values_mirror = Kokkos::create_mirror(values);
    Kokkos::deep_copy(values_mirror, values);
    ASSERT_EQ(values_mirror(0), 3.);
}

auto createDenseMatrix_3x3() {
    auto dense = Kokkos::View<double[3][3]>("dense");
    auto dense_host_data = std::array{1., 0., 0., 0., 2., 0., 0., 0., 3.};
    auto dense_host = Kokkos::View<double[3][3], Kokkos::HostSpace>(dense_host_data.data());
    auto dense_mirror = Kokkos::create_mirror(dense);
    Kokkos::deep_copy(dense_mirror, dense_host);
    Kokkos::deep_copy(dense, dense_mirror);
    return dense;
}

auto createRowPtrs_3x3() {
    auto row_ptrs = Kokkos::View<int[4]>("row_ptrs");
    auto row_ptrs_host_data = std::array{0, 1, 2, 3};
    auto row_ptrs_host = Kokkos::View<int[4], Kokkos::HostSpace>(row_ptrs_host_data.data());
    Kokkos::deep_copy(row_ptrs, row_ptrs_host);
    return row_ptrs;
}

auto createIndices_3x3() {
    auto indices = Kokkos::View<int[3]>("indices");
    auto indices_host_data = std::array{0, 1, 2};
    auto indices_host = Kokkos::View<int[3], Kokkos::HostSpace>(indices_host_data.data());
    Kokkos::deep_copy(indices, indices_host);
    return indices;
}

TEST(CopyIntoSparseMatrix, Diagonal) {
    using device_type =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device_type, void, int>;

    constexpr auto num_rows = 3;
    constexpr auto num_columns = 3;
    constexpr auto num_non_zero = 3;

    auto dense = createDenseMatrix_3x3();

    auto values = Kokkos::View<double[num_non_zero]>("values");
    auto row_ptrs = createRowPtrs_3x3();
    auto indices = createIndices_3x3();
    auto sparse =
        crs_matrix_type("sparse", num_rows, num_columns, num_non_zero, values, row_ptrs, indices);

    auto row_data_size = Kokkos::View<double*>::shmem_size(num_columns);
    auto col_idx_size = Kokkos::View<int*>::shmem_size(num_columns);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(num_rows, Kokkos::AUTO());
    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", sparse_matrix_policy,
        CopyIntoSparseMatrix<crs_matrix_type>{sparse, dense}
    );

    auto values_mirror = Kokkos::create_mirror(values);
    Kokkos::deep_copy(values_mirror, values);
    ASSERT_EQ(values_mirror(0), 1.);
    ASSERT_EQ(values_mirror(1), 2.);
    ASSERT_EQ(values_mirror(2), 3.);
}

auto createDenseMatrix_5x5() {
    auto dense = Kokkos::View<double[5][5]>("dense");
    auto dense_host_data = std::array{1.,  2.,  3., 00., 00., 4.,  5.,  6., 00., 00., 7.,  8., 9.,
                                      00., 00., 0., 0.,  0.,  10., 11., 0., 0.,  0.,  12., 13.};
    auto dense_host = Kokkos::View<double[5][5], Kokkos::HostSpace>(dense_host_data.data());
    auto dense_mirror = Kokkos::create_mirror(dense);
    Kokkos::deep_copy(dense_mirror, dense_host);
    Kokkos::deep_copy(dense, dense_mirror);
    return dense;
}

auto createRowPtrs_5x5() {
    auto row_ptrs = Kokkos::View<int[6]>("row_ptrs");
    auto row_ptrs_host_data = std::array{0, 3, 6, 9, 11, 13};
    auto row_ptrs_host = Kokkos::View<int[6], Kokkos::HostSpace>(row_ptrs_host_data.data());
    Kokkos::deep_copy(row_ptrs, row_ptrs_host);
    return row_ptrs;
}

auto createIndices_5x5() {
    auto indices = Kokkos::View<int[13]>("indices");
    auto indices_host_data = std::array{0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4};
    auto indices_host = Kokkos::View<int[13], Kokkos::HostSpace>(indices_host_data.data());
    Kokkos::deep_copy(indices, indices_host);
    return indices;
}

TEST(CopyIntoSparseMatrix, Block) {
    using device_type =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device_type, void, int>;

    constexpr auto num_rows = 5;
    constexpr auto num_columns = 5;
    constexpr auto num_non_zero = 13;

    auto dense = createDenseMatrix_5x5();

    auto values = Kokkos::View<double[num_non_zero]>("values");
    auto row_ptrs = createRowPtrs_5x5();
    auto indices = createIndices_5x5();
    auto sparse =
        crs_matrix_type("sparse", num_rows, num_columns, num_non_zero, values, row_ptrs, indices);

    auto row_data_size = Kokkos::View<double*>::shmem_size(num_columns);
    auto col_idx_size = Kokkos::View<int*>::shmem_size(num_columns);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(num_rows, Kokkos::AUTO());
    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for(
        "CopyIntoSparseMatrix", sparse_matrix_policy,
        CopyIntoSparseMatrix<crs_matrix_type>{sparse, dense}
    );

    auto values_mirror = Kokkos::create_mirror(values);
    Kokkos::deep_copy(values_mirror, values);
    ASSERT_EQ(values_mirror(0), 1.);
    ASSERT_EQ(values_mirror(1), 2.);
    ASSERT_EQ(values_mirror(2), 3.);
    ASSERT_EQ(values_mirror(3), 4.);
    ASSERT_EQ(values_mirror(4), 5.);
    ASSERT_EQ(values_mirror(5), 6.);
    ASSERT_EQ(values_mirror(6), 7.);
    ASSERT_EQ(values_mirror(7), 8.);
    ASSERT_EQ(values_mirror(8), 9.);
    ASSERT_EQ(values_mirror(9), 10.);
    ASSERT_EQ(values_mirror(10), 11.);
    ASSERT_EQ(values_mirror(11), 12.);
    ASSERT_EQ(values_mirror(12), 13.);
}

}  // namespace openturbine::tests
