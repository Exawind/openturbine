#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

struct ContributeLambdaToVector {
    Kokkos::View<FreedomSignature*>::const_type base_node_freedom_signature;
    Kokkos::View<FreedomSignature*>::const_type target_node_freedom_signature;

    Kokkos::View<size_t* [6]>::const_type base_node_freedom_table;
    Kokkos::View<size_t* [6]>::const_type target_node_freedom_table;

    Kokkos::View<double* [6]>::const_type base_lambda_residual_terms;
    Kokkos::View<double* [6]>::const_type target_lambda_residual_terms;

    Kokkos::View<double*> R;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
            const auto base_num_dofs = count_active_dofs(base_node_freedom_signature(i_constraint));
            const auto first_base_dof = base_node_freedom_table(i_constraint, 0);
            for (auto i_dof = 0U; i_dof < base_num_dofs; ++i_dof) {
                Kokkos::atomic_add(&R(first_base_dof + i_dof), base_lambda_residual_terms(i_constraint, i_dof));
            }

            const auto target_num_dofs = count_active_dofs(target_node_freedom_signature(i_constraint));
            const auto first_target_dof = target_node_freedom_table(i_constraint, 0);
            for (auto i_dof = 0U; i_dof < target_num_dofs; ++i_dof) {
                Kokkos::atomic_add(&R(first_target_dof + i_dof), target_lambda_residual_terms(i_constraint, i_dof));
            }
    }

};

}
