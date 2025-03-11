#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "compute_b_num_non_zero.hpp"
#include "populate_sparse_row_ptrs_col_inds_constraints.hpp"

namespace openturbine {

template <typename CrsMatrixType>
[[nodiscard]] inline CrsMatrixType CreateConstraintsMatrixFull(
    size_t num_system_dofs,
    size_t num_dofs,
    size_t num_constraint_dofs,
    const Kokkos::View<ConstraintType*>::const_type& constraint_type,
    const Kokkos::View<FreedomSignature*>::const_type& base_node_freedom_signature,
    const Kokkos::View<FreedomSignature*>::const_type& target_node_freedom_signature,
    const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
    const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
) {
    auto region = Kokkos::Profiling::ScopedRegion("Create Full Constraints Matrix");

    using ValuesType = typename CrsMatrixType::values_type::non_const_type;
    using RowPtrType = typename CrsMatrixType::staticcrsgraph_type::row_map_type::non_const_type;
    using IndicesType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    const auto B_num_rows = num_constraint_dofs;
    const auto B_num_non_zero = ComputeBNumNonZero(
        constraint_row_range, base_node_freedom_signature, target_node_freedom_signature
    );

    const auto B_row_ptrs = RowPtrType("b_row_ptrs", B_num_rows + 1);
    const auto B_col_ind = IndicesType("b_indices", B_num_non_zero);
    Kokkos::parallel_for(
        "PopulateSparseRowPtrsColInds_Constraints", 1,
        PopulateSparseRowPtrsColInds_Constraints<RowPtrType, IndicesType>{
            constraint_type, constraint_base_node_freedom_table,
            constraint_target_node_freedom_table, constraint_row_range, base_node_freedom_signature,
            target_node_freedom_signature, B_row_ptrs, B_col_ind
        }
    );

    const auto B_values = ValuesType("B values", B_num_non_zero);
    KokkosSparse::sort_crs_matrix(B_row_ptrs, B_col_ind, B_values);

    auto constraints_matrix_full_row_ptrs =
        RowPtrType("constraints_matrix_full_row_ptrs", num_dofs + 1);
    Kokkos::deep_copy(
        Kokkos::subview(
            constraints_matrix_full_row_ptrs, Kokkos::pair(num_system_dofs, num_dofs + 1)
        ),
        B_row_ptrs
    );

    return CrsMatrixType(
        "constraints_matrix_full", static_cast<int>(num_dofs), static_cast<int>(num_dofs),
        B_num_non_zero, B_values, constraints_matrix_full_row_ptrs,
        B_col_ind
    );
}

}  // namespace openturbine
