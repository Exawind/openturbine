#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct ContributeElementsToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    CrsMatrixType sparse;
    Kokkos::View<double*** [6][6]>::const_type dense;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i = member.league_rank();
        const auto row = sparse.row(i);
        const auto row_map = sparse.graph.row_map;
        const auto cols = sparse.graph.entries;
        const auto row_data = RowDataType(member.team_scratch(1), static_cast<size_t>(row.length));
        const auto col_idx = ColIdxType(member.team_scratch(1), static_cast<size_t>(row.length));
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [&](size_t entry) {
            const auto element = i / row.length;
            const auto row_block = i % row.length;
            const auto node_1 = row_block / 6;
            const auto component_1 = row_block % 6;
            const auto node_2 = entry / 6;
            const auto component_2 = entry % 6;
            col_idx(entry) = cols(row_map(i) + entry);
            row_data(entry) = dense(element, node_1, node_2, component_1, component_2);
        });
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
            sparse.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    }
};
}  // namespace openturbine
