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
        constexpr auto force_atomic = false;
        constexpr auto dofs_per_node = 3U;  // Springs only have translational DOFs
        constexpr auto total_dofs = 6U;     // 3 DOFs * 2 nodes

        auto row_data_data = Kokkos::Array<typename RowDataType::value_type, total_dofs>{};
        auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, total_dofs>{};
        auto row_data = RowDataType(row_data_data.data(), total_dofs);
        auto col_idx = ColIdxType(col_idx_data.data(), total_dofs);

        for (auto i_node = 0U; i_node < 2U; ++i_node) {
            for (auto j = 0U; j < dofs_per_node; ++j) {
                col_idx(i_node * dofs_per_node + j) =
                    static_cast<int>(element_freedom_table(i_elem, i_node, j));
            }
        }

        for (auto i_node_1 = 0U; i_node_1 < 2U; ++i_node_1) {
            for (auto i_dof = 0U; i_dof < dofs_per_node; ++i_dof) {
                const auto row_num = element_freedom_table(i_elem, i_node_1, i_dof);
                for (auto i_node_2 = 0U; i_node_2 < 2U; ++i_node_2) {
                    for (auto j_dof = 0U; j_dof < dofs_per_node; ++j_dof) {
                        row_data(i_node_2 * dofs_per_node + j_dof) =
                            dense(i_elem, i_node_1, i_node_2, i_dof, j_dof);
                    }
                }
                sparse.sumIntoValues(
                    static_cast<int>(row_num), col_idx.data(), total_dofs, row_data.data(),
                    is_sorted, force_atomic
                );
            }
        }
    }
};

}  // namespace openturbine
