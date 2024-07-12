#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct PopulateSparseRowPtrsColInds_Transpose {
    int rows;
    int cols;
    Kokkos::View<int*> col_count;       // cols
    Kokkos::View<int*> temp_row_ptr;    // cols + 1
    Kokkos::View<int*> row_ptrs;        // rows + 1
    Kokkos::View<int*> col_inds;        // nnz
    Kokkos::View<int*> row_ptrs_trans;  // cols + 1
    Kokkos::View<int*> col_inds_trans;  // nnz

    KOKKOS_FUNCTION
    void operator()(int) const {
        // Step 1: Count the non-zero entries for each column
        for (int i = 0; i < col_inds.extent_int(0); ++i) {
            col_count(col_inds(i))++;
        }

        // Step 2: Calculate the row pointers for the transposed matrix
        for (int i = 1; i <= cols; ++i) {
            row_ptrs_trans(i) = row_ptrs_trans(i - 1) + col_count(i - 1);
        }

        // Step 3: Initialize column indices for the transposed matrix
        // Copy of row pointers for writing indices
        for (int i = 0; i < row_ptrs_trans.extent_int(0); ++i) {
            temp_row_ptr(i) = row_ptrs_trans(i);
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = row_ptrs(i); j < row_ptrs(i + 1); ++j) {
                int col = col_inds(j);
                int dest_pos = temp_row_ptr(col)++;
                col_inds_trans(dest_pos) = i;
            }
        }
    }
};

}  // namespace openturbine
