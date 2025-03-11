#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "fill_unshifted_row_ptrs.hpp"

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateSystemMatrixFull(
    size_t num_system_dofs, size_t num_dofs, const CrsMatrixType& system_matrix
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full System Matrix");

    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;

    auto system_matrix_full_row_ptrs = RowPtrType("system_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::parallel_for(
        "FillUnshiftedRowPtrs", num_dofs + 1,
        FillUnshiftedRowPtrs<RowPtrType>{
            num_system_dofs, system_matrix.graph.row_map, system_matrix_full_row_ptrs
        }
    );
    return CrsMatrixType(
        "system_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
        system_matrix.nnz(), system_matrix.values, system_matrix_full_row_ptrs,
        system_matrix.graph.entries
    );
}

}  // namespace openturbine
