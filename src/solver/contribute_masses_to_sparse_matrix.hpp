#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename CrsMatrixType>
struct ContributeMassesToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    Kokkos::View<FreedomSignature*> element_freedom_signature;
    Kokkos::View<size_t* [6]> element_freedom_table;
    Kokkos::View<double* [6][6]>::const_type dense;
    CrsMatrixType sparse;

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        constexpr auto is_sorted = true;
        constexpr auto force_atomic = false;
        const auto num_dofs = count_active_dofs(element_freedom_signature(i));
        auto row_data_data = Kokkos::Array<typename RowDataType::value_type, 6>{};
        auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, 6>{};
        auto row_data = RowDataType(row_data_data.data(), num_dofs);
        auto col_idx = ColIdxType(col_idx_data.data(), num_dofs);

        for (auto component_2 = 0U; component_2 < num_dofs; ++component_2) {
            col_idx(component_2) = static_cast<int>(element_freedom_table(i, component_2));
        }
        for (auto component_1 = 0U; component_1 < num_dofs; ++component_1) {
            const auto row_num = element_freedom_table(i, component_1);
            for (auto component_2 = 0U; component_2 < num_dofs; ++component_2) {
                row_data(component_2) = dense(i, component_1, component_2);
            }
            sparse.sumIntoValues(
                static_cast<int>(row_num), col_idx.data(), static_cast<int>(num_dofs),
                row_data.data(), is_sorted, force_atomic
            );
        }
    }
};

}  // namespace openturbine
