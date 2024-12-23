#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateConstraintsMatrixFull(
    size_t num_system_dofs, size_t num_dofs, const CrsMatrixType& constraints_matrix
) {
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    auto constraints_matrix_full_row_ptrs =
        RowPtrType("constraints_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::deep_copy(
        Kokkos::subview(
            constraints_matrix_full_row_ptrs, Kokkos::pair(num_system_dofs, num_dofs + 1)
        ),
        constraints_matrix.graph.row_map
    );
    return CrsMatrixType(
        "constraints_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
        constraints_matrix.nnz(), constraints_matrix.values, constraints_matrix_full_row_ptrs,
        constraints_matrix.graph.entries
    );
}

}  // namespace openturbine
