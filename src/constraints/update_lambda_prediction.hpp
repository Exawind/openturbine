#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct UpdateLambdaPrediction {
    size_t num_system_dofs;
    typename Kokkos::View<Kokkos::pair<size_t, size_t>*, DeviceType>::const_type row_range;
    typename Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType>::const_type x;
    Kokkos::View<double* [6], DeviceType> lambda;

    KOKKOS_FUNCTION
    void operator()(size_t constraint) const {
        const auto first_index = row_range(constraint).first;
        const auto max_index = row_range(constraint).second;
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;
        for (auto row = first_index; row < max_index; ++row) {
            if constexpr (force_atomic) {
                Kokkos::atomic_add(
                    &lambda(constraint, row - first_index), x(num_system_dofs + row, 0)
                );
            } else {
                lambda(constraint, row - first_index) += x(num_system_dofs + row, 0);
            }
        }
    }
};

}  // namespace openturbine
