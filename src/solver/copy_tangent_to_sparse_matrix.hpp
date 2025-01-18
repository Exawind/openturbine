#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {
template <typename CrsMatrixType>
struct CopyTangentToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    Kokkos::View<size_t*>::const_type ID;
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<double* [6][6]>::const_type dense;
    CrsMatrixType sparse;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i = member.league_rank();
        const auto node_id = ID(i);
        const auto num_dofs = count_active_dofs(node_freedom_allocation_table(node_id));
        const auto first_entry = node_freedom_map_table(node_id);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_dofs), [&](size_t dof) {
            const auto row_num = first_entry + dof;
            auto row_map = sparse.graph.row_map;
            auto cols = sparse.graph.entries;
            auto row_data_data = Kokkos::Array<typename RowDataType::value_type, 6>{};
            auto col_idx_data = Kokkos::Array<typename ColIdxType::value_type, 6>{};
            auto row_data = RowDataType(row_data_data.data(), num_dofs);
            auto col_idx = ColIdxType(col_idx_data.data(), num_dofs);

            for (auto entry = 0U; entry < num_dofs; ++entry) {
                col_idx(entry) = cols(row_map(row_num) + entry);
                row_data(entry) = dense(node_id, dof, entry);
            }
            sparse.replaceValues(
                static_cast<int>(row_num), col_idx.data(), static_cast<int>(num_dofs),
                row_data.data()
            );
        });
    }
};
}  // namespace openturbine
