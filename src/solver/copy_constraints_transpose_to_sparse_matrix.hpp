#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopyConstraintsTransposeToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type base_node_col_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;
    Kokkos::View<FreedomSignature*> base_node_freedom_signature;
    Kokkos::View<FreedomSignature*> target_node_freedom_signature;
    Kokkos::View<size_t* [6]>::const_type base_node_freedom_table;
    Kokkos::View<size_t* [6]>::const_type target_node_freedom_table;
    Kokkos::View<double* [6][6]>::const_type base_dense;
    Kokkos::View<double* [6][6]>::const_type target_dense;
    CrsMatrixType sparse;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_constraint = member.league_rank();
        constexpr bool is_sorted = true;
        const auto num_cols = row_range(i_constraint).second - row_range(i_constraint).first;
        const auto num_base_dofs = count_active_dofs(base_node_freedom_signature(i_constraint));
        const auto base_start_row = base_node_freedom_table(i_constraint, 0);
        const auto base_end_row = base_start_row + num_base_dofs;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, base_start_row, base_end_row), [&](size_t i) {
            const auto row_number = i - base_start_row;
            auto row_data_data = Kokkos::Array<typename RowDataType::value_type, 6>{};
            auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, 6>{};
            const auto row_data = RowDataType(row_data_data.data(), num_cols);
            const auto col_idx = ColIdxType(col_idx_data.data(), num_cols);
            for (auto entry = 0U; entry < num_cols; ++entry) {
                col_idx(entry) = row_range(i_constraint).first + entry;
            }
            for (auto entry = 0U; entry < num_cols; ++entry) {
                row_data(entry) = base_dense(i_constraint, row_number, entry);
            }
            sparse.replaceValues(i, col_idx.data(), num_cols, row_data.data(), is_sorted);
        });

        const auto num_target_dofs = count_active_dofs(target_node_freedom_signature(i_constraint));
        const auto target_start_row = target_node_freedom_table(i_constraint, 0);
        const auto target_end_row = target_start_row + num_target_dofs;;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, target_start_row, target_end_row), [&](size_t i) {
            const auto row_number = i - target_start_row;
            auto row_data_data = Kokkos::Array<typename RowDataType::value_type, 6>{};
            auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, 6>{};
            const auto row_data = RowDataType(row_data_data.data(), num_cols);
            const auto col_idx = ColIdxType(col_idx_data.data(), num_cols);
            for (auto entry = 0U; entry < num_cols; ++entry) {
                col_idx(entry) = row_range(i_constraint).first + entry;
            }
            for (auto entry = 0U; entry < num_cols; ++entry) {
                row_data(entry) = target_dense(i_constraint, row_number, entry);
            }

            sparse.replaceValues(i, col_idx.data(), num_cols, row_data.data(), is_sorted);
        });
    }
};
}
