#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopyIntoSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;

    CrsMatrixType sparse;
    Kokkos::View<const double**> dense;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        auto i = member.league_rank();
        auto row = sparse.row(i);
        auto row_map = sparse.graph.row_map;
        auto cols = sparse.graph.entries;
        auto row_data = RowDataType(member.team_scratch(1), static_cast<size_t>(row.length));
        auto col_idx = ColIdxType(member.team_scratch(1), static_cast<size_t>(row.length));
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [&](size_t entry) {
            col_idx(entry) = cols(static_cast<size_t>(row_map(i)) + entry);
            row_data(entry) = dense(i, col_idx(entry));
        });
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
            sparse.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    }
};
}  // namespace openturbine
