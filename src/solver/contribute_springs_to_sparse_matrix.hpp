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

        for (auto node_1 = 0U; node_1 < 2U; ++node_1) {
            auto row_data_data = Kokkos::Array<typename RowDataType::value_type, 3U>{};
            auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, 3U>{};
            auto row_data = RowDataType(row_data_data.data(), 3U);
            auto col_idx = ColIdxType(col_idx_data.data(), 3U);

            for (auto node_2 = 0U; node_2 < 2U; ++node_2) {
                for (auto component_2 = 0U; component_2 < 3U; ++component_2) {
                    col_idx(component_2) =
                        static_cast<int>(element_freedom_table(i_elem, node_2, component_2));
                }
                for (auto component_1 = 0U; component_1 < 3U; ++component_1) {
                    const auto row_num = element_freedom_table(i_elem, node_1, component_1);
                    for (auto component_2 = 0U; component_2 < 3U; ++component_2) {
                        row_data(component_2) =
                            dense(i_elem, node_1, node_2, component_1, component_2);
                    }
                    sparse.sumIntoValues(
                        static_cast<int>(row_num), col_idx.data(), 3, row_data.data(), is_sorted,
                        force_atomic
                    );
                }
            }
        }
    }
};

}  // namespace openturbine
