#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopyConstraintsToSparseMatrix {
    using DeviceType = typename CrsMatrixType::device_type;
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    size_t num_system_rows{};
    typename Kokkos::View<Kokkos::pair<size_t, size_t>*, DeviceType>::const_type row_range;
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type base_node_freedom_signature;
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type target_node_freedom_signature;
    typename Kokkos::View<size_t* [6], DeviceType>::const_type base_node_freedom_table;
    typename Kokkos::View<size_t* [6], DeviceType>::const_type target_node_freedom_table;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type base_dense;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type target_dense;
    CrsMatrixType sparse;

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        const auto i_constraint = member.league_rank();
        constexpr auto is_sorted = true;
        const auto start_row = row_range(i_constraint).first;
        const auto end_row = row_range(i_constraint).second;
        const auto num_base_dofs = count_active_dofs(base_node_freedom_signature(i_constraint));
        const auto num_target_dofs = count_active_dofs(target_node_freedom_signature(i_constraint));

        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, start_row, end_row), [&](size_t i) {
            const auto row_number = i - start_row;
            const auto real_row = num_system_rows + i;
            constexpr auto hint = static_cast<typename CrsMatrixType::ordinal_type>(0);
            auto row = sparse.row(static_cast<int>(real_row));
            auto first_col = static_cast<typename CrsMatrixType::ordinal_type>(
                base_node_freedom_table(i_constraint, 0)
            );
            auto offset = KokkosSparse::findRelOffset(
                &(row.colidx(0)), row.length, first_col, hint, is_sorted
            );
            for (auto entry = 0U; entry < num_base_dofs; ++entry, ++offset) {
                row.value(offset) = base_dense(i_constraint, row_number, entry);
            }

            first_col = static_cast<typename CrsMatrixType::ordinal_type>(
                target_node_freedom_table(i_constraint, 0)
            );
            offset = KokkosSparse::findRelOffset(
                &(row.colidx(0)), row.length, first_col, hint, is_sorted
            );
            for (auto entry = 0U; entry < num_target_dofs; ++entry, ++offset) {
                row.value(offset) = target_dense(i_constraint, row_number, entry);
            }
        });
    }
};
}  // namespace openturbine
