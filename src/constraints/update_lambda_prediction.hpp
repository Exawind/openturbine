#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename DeviceType>
struct UpdateLambdaPrediction {
    size_t num_system_dofs;
    typename Kokkos::View<Kokkos::pair<size_t, size_t>*, DeviceType>::const_type row_range;
    typename Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType>::const_type x;
    Kokkos::View<double* [6], DeviceType> lambda;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
        const auto first_index = row_range(i_constraint).first;
        const auto max_index = row_range(i_constraint).second;
        for (auto row = first_index; row < max_index; ++row) {
            lambda(i_constraint, row - first_index) += x(num_system_dofs + row, 0);
        }
    }
};

}  // namespace openturbine
