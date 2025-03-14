#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename RowPtrType>
struct ComputeConstraintsRowEntries {
    using ValueType = typename RowPtrType::value_type;
    size_t num_system_dofs;
    Kokkos::View<size_t*>::const_type base_active_dofs;
    Kokkos::View<size_t*>::const_type target_active_dofs;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    RowPtrType row_entries;

    KOKKOS_FUNCTION
    void operator()(size_t constraint) const {
        const auto first_row = row_range(constraint).first;
        const auto end_row = row_range(constraint).second;
        const auto num_base = static_cast<ValueType>(base_active_dofs(constraint));
        const auto num_target = static_cast<ValueType>(target_active_dofs(constraint));
        for (auto row = first_row; row < end_row; ++row) {
            row_entries(row + num_system_dofs) = num_base + num_target;
        }
    }
};

}
