#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

struct UpdateDynamicPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<double* [1], Kokkos::LayoutLeft>::const_type x_delta;
    Kokkos::View<double* [6]> q_delta;
    Kokkos::View<double* [6]> v;
    Kokkos::View<double* [6]> vd;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        const auto num_dof = count_active_dofs(node_freedom_allocation_table(i_node));
        const auto first_dof = node_freedom_map_table(i_node);
        for (auto j = 0U; j < num_dof; ++j) {
            const auto delta = x_delta(first_dof + j, 0);
            q_delta(i_node, j) += delta / h;
            v(i_node, j) += gamma_prime * delta;
            vd(i_node, j) += beta_prime * delta;
        }
    }
};

}  // namespace openturbine
