#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename DeviceType>
struct UpdateDynamicPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type node_freedom_allocation_table;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_freedom_map_table;
    typename Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType>::const_type x_delta;
    Kokkos::View<double* [6], DeviceType> q_delta;
    Kokkos::View<double* [6], DeviceType> v;
    Kokkos::View<double* [6], DeviceType> vd;

    KOKKOS_FUNCTION
    void operator()(const size_t node) const {
        const auto num_dof = count_active_dofs(node_freedom_allocation_table(node));
        const auto first_dof = node_freedom_map_table(node);
        for (auto component = 0U; component < num_dof; ++component) {
            const auto delta = x_delta(first_dof + component, 0);
            q_delta(node, component) += delta / h;
            v(node, component) += gamma_prime * delta;
            vd(node, component) += beta_prime * delta;
        }
    }
};

}  // namespace openturbine
