#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {
template <typename CrsMatrixType>
struct CopyConstraintsTransposeToSparseMatrix {
    using DeviceType = typename CrsMatrixType::device_type;
    using RowDataType = typename CrsMatrixType::values_type::non_const_type;
    using ColIdxType = typename CrsMatrixType::staticcrsgraph_type::entries_type::non_const_type;
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    size_t num_system_cols{};
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
        constexpr bool is_sorted = true;
        const auto num_cols = static_cast<int>(row_range(i_constraint).second - row_range(i_constraint).first);
        const auto first_col = static_cast<int>(row_range(i_constraint).first + num_system_cols);
        const auto num_base_dofs = static_cast<int>(count_active_dofs(base_node_freedom_signature(i_constraint)));
        const auto base_start_row = static_cast<int>(base_node_freedom_table(i_constraint, 0));
        const auto base_end_row = base_start_row + num_base_dofs;
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, base_start_row, base_end_row),
            [&](int i) {
                const auto row_number = i - base_start_row;
                const auto hint = 0;
                auto row = sparse.row(i);

                auto offset = KokkosSparse::findRelOffset(&(row.colidx(0)), row.length, first_col, hint, is_sorted);
                for (auto entry = 0; entry < num_cols; ++entry, ++offset) {
                    row.value(offset) = base_dense(i_constraint, row_number, entry);
                }
            }
        );

        const auto num_target_dofs = static_cast<int>(count_active_dofs(target_node_freedom_signature(i_constraint)));
        const auto target_start_row = static_cast<int>(target_node_freedom_table(i_constraint, 0));
        const auto target_end_row = target_start_row + num_target_dofs;
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, target_start_row, target_end_row),
            [&](int i) {
                const auto row_number = i - target_start_row;
                const auto hint = 0;
                auto row = sparse.row(i);

                auto offset = KokkosSparse::findRelOffset(&(row.colidx(0)), row.length, first_col, hint, is_sorted);
                for (auto entry = 0; entry < num_cols; ++entry, ++offset) {
                    row.value(offset) = target_dense(i_constraint, row_number, entry);
                }
            }
        );
    }
};
}  // namespace openturbine
