#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/constraints/constraint_type.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

template <typename RowPtrType, typename IndicesType>
struct PopulateSparseRowPtrsColInds_Transpose {
    size_t rows;
    size_t cols;
    typename RowPtrType::const_type row_ptrs;   // rows + 1
    typename IndicesType::const_type col_inds;  // nnz
    IndicesType col_count;                      // cols
    RowPtrType temp_row_ptr;                    // cols + 1
    RowPtrType row_ptrs_trans;                  // cols + 1
    IndicesType col_inds_trans;                 // nnz

    KOKKOS_FUNCTION
    void operator()(int) const {
        // Step 1: Count the non-zero entries for each column
        for (auto i = 0U; i < col_inds.extent(0); ++i) {
            col_count(col_inds(i))++;
        }

        // Step 2: Calculate the row pointers for the transposed matrix
        for (auto i = 1U; i <= cols; ++i) {
            row_ptrs_trans(i) = row_ptrs_trans(i - 1) +
                                static_cast<typename RowPtrType::value_type>(col_count(i - 1));
        }

        // Step 3: Initialize column indices for the transposed matrix
        // Copy of row pointers for writing indices
        for (auto i = 0U; i < row_ptrs_trans.extent(0); ++i) {
            temp_row_ptr(i) = row_ptrs_trans(i);
        }

        for (auto i = 0U; i < rows; ++i) {
            for (auto j = row_ptrs(i); j < row_ptrs(i + 1); ++j) {
                auto col = col_inds(j);
                auto dest_pos = temp_row_ptr(col)++;
                col_inds_trans(dest_pos) = static_cast<typename IndicesType::value_type>(i);
            }
        }
    }
};

}  // namespace openturbine
