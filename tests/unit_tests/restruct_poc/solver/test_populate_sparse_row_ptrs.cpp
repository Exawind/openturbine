#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/restruct_poc/solver/populate_sparse_row_ptrs.hpp"

namespace openturbine::restruct_poc::tests {

TEST(PopulateSparseRowPtrs, SingleElement) {
    auto elem_indices_host = Kokkos::View<Beams::ElemIndices[1], Kokkos::HostSpace>("elem_indices_host");
    auto elem_indices = Kokkos::View<Beams::ElemIndices[1]>("elem_indices");
    elem_indices_host(0).num_nodes = 5;
    Kokkos::deep_copy(elem_indices, elem_indices_host);

    constexpr auto num_rows = 5*6;
    auto row_ptrs = Kokkos::View<int[num_rows+1]>("row_ptrs");

    Kokkos::parallel_for(1, PopulateSparseRowPtrs{elem_indices, row_ptrs});
    
    auto row_ptrs_host = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_host, row_ptrs);
    for(int row = 0; row < num_rows+1; ++row) {
      EXPECT_EQ(row_ptrs_host(row), num_rows*row);
    }
    
}

TEST(PopulateSparseRowPtrs, TwoElements) {
    auto elem_indices_host = Kokkos::View<Beams::ElemIndices[2], Kokkos::HostSpace>("elem_indices_host");
    auto elem_indices = Kokkos::View<Beams::ElemIndices[2]>("elem_indices");
    elem_indices_host(0).num_nodes = 5;
    elem_indices_host(1).num_nodes = 3;
    Kokkos::deep_copy(elem_indices, elem_indices_host);

    constexpr auto elem1_rows = 5*6;
    constexpr auto elem2_rows = 3*6;
    constexpr auto num_rows = elem1_rows + elem2_rows;
    auto row_ptrs = Kokkos::View<int[num_rows+1]>("row_ptrs");

    Kokkos::parallel_for(1, PopulateSparseRowPtrs{elem_indices, row_ptrs});
    
    auto row_ptrs_host = Kokkos::create_mirror(row_ptrs);
    Kokkos::deep_copy(row_ptrs_host, row_ptrs);
    for(int row = 0; row < elem1_rows+1; ++row) {
      EXPECT_EQ(row_ptrs_host(row), elem1_rows*row);
    }
    for(int row = elem1_rows*(elem1_rows+1); row < elem1_rows+1+elem2_rows; ++row) {
        EXPECT_EQ(row_ptrs_host(row), elem1_rows*(elem1_rows+1) + elem2_rows*(row - elem1_rows*(elem1_rows+1)));
    }
}

}