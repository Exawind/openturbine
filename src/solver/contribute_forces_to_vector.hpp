#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

template <typename DeviceType>
struct ContributeForcesToVector {
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type node_freedom_allocation_table;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_freedom_map_table;
    typename Kokkos::View<double**, DeviceType>::const_type node_loads;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> vector;

    KOKKOS_FUNCTION
    void operator()(size_t node) const {
        const auto this_node_freedom_signature = node_freedom_allocation_table(node);
        const auto dof_index = node_freedom_map_table(node);
        if (this_node_freedom_signature == FreedomSignature::NoComponents) {
            return;
        } else if (this_node_freedom_signature == FreedomSignature::AllComponents) {
            vector(dof_index + 0, 0) -= node_loads(node, 0);
            vector(dof_index + 1, 0) -= node_loads(node, 1);
            vector(dof_index + 2, 0) -= node_loads(node, 2);
            vector(dof_index + 3, 0) -= node_loads(node, 3);
            vector(dof_index + 4, 0) -= node_loads(node, 4);
            vector(dof_index + 5, 0) -= node_loads(node, 5);
        } else if (this_node_freedom_signature == FreedomSignature::JustPosition) {
            vector(dof_index + 0, 0) -= node_loads(node, 0);
            vector(dof_index + 1, 0) -= node_loads(node, 1);
            vector(dof_index + 2, 0) -= node_loads(node, 2);
        } else if (this_node_freedom_signature == FreedomSignature::JustRotation) {
            vector(dof_index + 0, 0) -= node_loads(node, 3);
            vector(dof_index + 1, 0) -= node_loads(node, 4);
            vector(dof_index + 2, 0) -= node_loads(node, 5);
        }
    }
};

}  // namespace openturbine
