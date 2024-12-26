#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "src/dof_management/freedom_signature.hpp"

namespace openturbine {
template <typename CrsMatrixType>
struct ContributeSpringsToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    Kokkos::View<FreedomSignature* [2]> element_freedom_signature;
    Kokkos::View<size_t* [2][3]> element_freedom_table;
    Kokkos::View<double* [2][2][3][3]>::const_type dense;  //< Element Stiffness matrices
    CrsMatrixType sparse;                                  //< Global sparse stiffness matrix

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        constexpr auto is_sorted = true;
        constexpr auto force_atomic = !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>;
        constexpr auto dofs_per_node = 3U;  // Springs only have translational DOFs
        constexpr auto total_dofs = 6U;     // 3 DOFs * 2 nodes

        auto row_data_data = Kokkos::Array<typename RowDataType::value_type, total_dofs>{};
        auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, total_dofs>{};
        auto row_data_node_1 = RowDataType(row_data_data.data(), total_dofs);
        auto row_data_node_2 = RowDataType(row_data_data.data(), total_dofs);
        auto col_idx = ColIdxType(col_idx_data.data(), total_dofs);

        for (auto j = 0U; j < dofs_per_node; ++j) {
            col_idx(j) = static_cast<int>(element_freedom_table(i_elem, 0, j));
            col_idx(j + dofs_per_node) = static_cast<int>(element_freedom_table(i_elem, 1, j));
        }
        for (auto i_dof = 0U; i_dof < dofs_per_node; ++i_dof) {
            const auto row_num_node_1 = element_freedom_table(i_elem, 0, i_dof);
            const auto row_num_node_2 = element_freedom_table(i_elem, 1, i_dof);
            for (auto j_dof = 0U; j_dof < dofs_per_node; ++j_dof) {
                row_data_node_1(j_dof) = dense(i_elem, 0, 0, i_dof, j_dof);
                row_data_node_1(dofs_per_node + j_dof) = dense(i_elem, 0, 1, i_dof, j_dof);
                row_data_node_2(j_dof) = dense(i_elem, 1, 0, i_dof, j_dof);
                row_data_node_2(dofs_per_node + j_dof) = dense(i_elem, 1, 1, i_dof, j_dof);
            }
            sparse.sumIntoValues(
                static_cast<int>(row_num_node_1), col_idx.data(), total_dofs, row_data_node_1.data(),
                is_sorted, force_atomic
            );
            sparse.sumIntoValues(
                static_cast<int>(row_num_node_2), col_idx.data(), total_dofs, row_data_node_2.data(),
                is_sorted, force_atomic
            );
        }
    }
};

}  // namespace openturbine
