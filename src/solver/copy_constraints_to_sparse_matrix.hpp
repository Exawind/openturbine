#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopyConstraintsToSparseMatrix {
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    size_t num_system_rows;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
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
        constexpr auto is_sorted = true;
        const auto start_row = row_range(i_constraint).first;
        const auto end_row = row_range(i_constraint).second;
        const auto num_base_dofs = count_active_dofs(base_node_freedom_signature(i_constraint));
        const auto num_target_dofs = count_active_dofs(target_node_freedom_signature(i_constraint));

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, start_row, end_row), [&](size_t i) {
            const auto row_number = i - start_row;
            const auto real_row = num_system_rows + i;
            auto row_data = Kokkos::Array<typename RowDataType::value_type, 6>{};
            auto col_idx = Kokkos::Array<typename ColIdxType::value_type, 6>{};
            for (auto entry = 0U; entry < num_base_dofs; ++entry) {
                col_idx[entry] = static_cast<int>(base_node_freedom_table(i_constraint, entry));
                row_data[entry] = base_dense(i_constraint, row_number, entry);
            }
            sparse.replaceValues(
                static_cast<int>(real_row), col_idx.data(), static_cast<int>(num_base_dofs),
                row_data.data(), is_sorted
            );
            for (auto entry = 0U; entry < num_target_dofs; ++entry) {
                col_idx[entry] = static_cast<int>(target_node_freedom_table(i_constraint, entry));
                row_data[entry] = target_dense(i_constraint, row_number, entry);
            }
            sparse.replaceValues(
                static_cast<int>(real_row), col_idx.data(), static_cast<int>(num_target_dofs),
                row_data.data(), is_sorted
            );
        });
    }
};
}  // namespace openturbine
