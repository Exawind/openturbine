#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

struct UpdateStaticPrediction {
    double h;
    double beta_prime;
    double gamma_prime;
    Kokkos::View<FreedomSignature*> node_freedom_allocation_table;
    Kokkos::View<size_t*> node_freedom_map_table;
    Kokkos::View<double*>::const_type x_delta;
    Kokkos::View<double* [6]> q_delta;

    KOKKOS_FUNCTION
    void operator()(const size_t i_node) const {
        const auto num_dof = count_active_dofs(node_freedom_allocation_table(i_node));
        const auto first_dof = node_freedom_map_table(i_node);
        for (auto j = 0U; j < num_dof; ++j) {
            const auto delta = x_delta(first_dof + j);
            q_delta(i_node, j) += delta / h;
        }
    }
};

}  // namespace openturbine
