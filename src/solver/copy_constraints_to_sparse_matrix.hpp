#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopyConstraintsToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type base_node_col_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;
    Kokkos::View<double* [6][6]>::const_type base_dense;
    Kokkos::View<double* [6][6]>::const_type target_dense;
    CrsMatrixType sparse;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_constraint = member.league_rank();
        const auto start_row = row_range(i_constraint).first;
        const auto end_row = row_range(i_constraint).second;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, start_row, end_row), [&](int i) {
            const auto row_number = static_cast<size_t>(i) - start_row;
            const auto row = sparse.row(i);
            const auto row_map = sparse.graph.row_map;
            const auto cols = sparse.graph.entries;
            const auto length = static_cast<size_t>(row.length);
            auto row_data_data = Kokkos::Array<typename RowDataType::value_type, 12>{};
            auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, 12>{};
            const auto row_data = RowDataType(row_data_data.data(), length);
            const auto col_idx = ColIdxType(col_idx_data.data(), length);
            for (auto entry = 0U; entry < length; ++entry) {
                col_idx(entry) = cols(row_map(i) + entry);
            }
            for (auto entry = base_node_col_range(i_constraint).first;
                 entry < base_node_col_range(i_constraint).second; ++entry) {
                row_data(entry) = base_dense(
                    i_constraint, row_number, entry - base_node_col_range(i_constraint).first
                );
            }
            for (auto entry = target_node_col_range(i_constraint).first;
                 entry < target_node_col_range(i_constraint).second; ++entry) {
                row_data(entry) = target_dense(
                    i_constraint, row_number, entry - target_node_col_range(i_constraint).first
                );
            }
            sparse.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    }
};
}  // namespace openturbine
