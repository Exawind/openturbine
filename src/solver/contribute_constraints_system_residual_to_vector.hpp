#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct ContributeConstraintsSystemResidualToVector {
    typename Kokkos::View<size_t* [6], DeviceType>::const_type target_node_freedom_table;
    typename Kokkos::View<size_t*, DeviceType>::const_type target_active_dofs;
    typename Kokkos::View<double* [6], DeviceType>::const_type system_residual_terms;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> residual;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
        const auto num_dofs = target_active_dofs(i_constraint);
        for (auto i = 0U; i < num_dofs; ++i) {
            residual(target_node_freedom_table(i_constraint, i), 0) +=
                system_residual_terms(i_constraint, i);
        }
    }
};

}  // namespace openturbine
