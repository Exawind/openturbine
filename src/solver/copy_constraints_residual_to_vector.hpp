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
    void operator()(size_t constraint) const {
        const auto first_row = row_range(constraint).first + start_row;
        const auto num_rows = row_range(constraint).second - row_range(constraint).first;
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;
        for (auto component = 0U; component < num_rows; ++component) {
            if constexpr (force_atomic) {
                Kokkos::atomic_add(
                    &residual(first_row + component, 0),
                    constraint_residual_terms(constraint, component)
                );
            } else {
                residual(first_row + component, 0) =
                    constraint_residual_terms(constraint, component);
            }
        }
    }
};
}  // namespace openturbine
