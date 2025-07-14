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
    void operator()(size_t constraint) const {
        const auto base_num_dofs = count_active_dofs(base_node_freedom_signature(constraint));
        const auto first_base_dof = base_node_freedom_table(constraint, 0);
        for (auto component = 0U; component < base_num_dofs; ++component) {
            Kokkos::atomic_add(
                &R(first_base_dof + component, 0), base_lambda_residual_terms(constraint, component)
            );
        }

        const auto target_num_dofs = count_active_dofs(target_node_freedom_signature(constraint));
        const auto first_target_dof = target_node_freedom_table(constraint, 0);
        for (auto component = 0U; component < target_num_dofs; ++component) {
            Kokkos::atomic_add(
                &R(first_target_dof + component, 0),
                target_lambda_residual_terms(constraint, component)
            );
        }
    }
};

}  // namespace openturbine
