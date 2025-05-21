#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename DeviceType>
struct ContributeLambdaToVector {
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type base_node_freedom_signature;
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type target_node_freedom_signature;

    typename Kokkos::View<size_t* [6], DeviceType>::const_type base_node_freedom_table;
    typename Kokkos::View<size_t* [6], DeviceType>::const_type target_node_freedom_table;

    typename Kokkos::View<double* [6], DeviceType>::const_type base_lambda_residual_terms;
    typename Kokkos::View<double* [6], DeviceType>::const_type target_lambda_residual_terms;

    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> R;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
        const auto base_num_dofs = count_active_dofs(base_node_freedom_signature(i_constraint));
        const auto first_base_dof = base_node_freedom_table(i_constraint, 0);
        for (auto i_dof = 0U; i_dof < base_num_dofs; ++i_dof) {
            Kokkos::atomic_add(
                &R(first_base_dof + i_dof, 0), base_lambda_residual_terms(i_constraint, i_dof)
            );
        }

        const auto target_num_dofs = count_active_dofs(target_node_freedom_signature(i_constraint));
        const auto first_target_dof = target_node_freedom_table(i_constraint, 0);
        for (auto i_dof = 0U; i_dof < target_num_dofs; ++i_dof) {
            Kokkos::atomic_add(
                &R(first_target_dof + i_dof, 0), target_lambda_residual_terms(i_constraint, i_dof)
            );
        }
    }
};

}  // namespace openturbine
