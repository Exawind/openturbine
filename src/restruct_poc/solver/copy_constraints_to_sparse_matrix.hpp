#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopyConstraintsToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    Kokkos::View<Constraints::DeviceData*>::const_type data;
    CrsMatrixType sparse;
    Kokkos::View<const double* [6][12]> dense;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        auto i_constraint = member.league_rank();
        auto& cd = data(i_constraint);
        auto start_row = cd.row_range.first;
        auto end_row = cd.row_range.second;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, start_row, end_row), [&](int i) {
            auto row_number = i - start_row;
            auto row = sparse.row(i);
            auto row_map = sparse.graph.row_map;
            auto cols = sparse.graph.entries;
            auto row_data_data = Kokkos::Array<typename RowDataType::value_type, 12>{};
            auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, 12>{};
            auto row_data = RowDataType(row_data_data.data(), row.length);
            auto col_idx = ColIdxType(col_idx_data.data(), row.length);
            for (int entry = 0; entry < row.length; ++entry) {
                col_idx(entry) = cols(row_map(i) + entry);
                row_data(entry) = dense(i_constraint, row_number, entry);
            }
            sparse.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    }
};
}  // namespace openturbine