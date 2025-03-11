#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "fill_unshifted_row_ptrs.hpp"

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateTransposeMatrixFull(
    size_t num_system_dofs, size_t num_dofs, const CrsMatrixType& B_t
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full Transpose Constraints Matrix");

    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    auto transpose_matrix_full_row_ptrs = RowPtrType("transpose_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::parallel_for(
        "FillUnshiftedRowPtrs", num_dofs + 1,
        FillUnshiftedRowPtrs<RowPtrType>{
            num_system_dofs, B_t.graph.row_map, transpose_matrix_full_row_ptrs
        }
    );
    auto transpose_matrix_full_indices = IndicesType("transpose_matrix_full_indices", B_t.nnz());
    Kokkos::deep_copy(transpose_matrix_full_indices, static_cast<int>(num_system_dofs));
    KokkosBlas::axpy(1, B_t.graph.entries, transpose_matrix_full_indices);
    return CrsMatrixType(
        "transpose_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs), B_t.nnz(),
        B_t.values, transpose_matrix_full_row_ptrs, transpose_matrix_full_indices
    );
}

}  // namespace openturbine
