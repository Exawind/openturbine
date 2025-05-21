#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct CopyConstraintsResidualToVector {
    size_t start_row;
    typename Kokkos::View<Kokkos::pair<size_t, size_t>*, DeviceType>::const_type row_range;
    typename Kokkos::View<double* [6], DeviceType>::const_type constraint_residual_terms;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> residual;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
        const auto first_row = row_range(i_constraint).first + start_row;
        const auto num_rows = row_range(i_constraint).second - row_range(i_constraint).first;
        for (auto i = 0U; i < num_rows; ++i) {
            residual(first_row + i, 0) = constraint_residual_terms(i_constraint, i);
        }
    }
};
}  // namespace openturbine
