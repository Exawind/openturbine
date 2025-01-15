#pragma once

#include <Kokkos_Core.hpp>

#include "src/dof_management/freedom_signature.hpp"

namespace openturbine {

struct ContributeForcesToVector {
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<double**>::const_type node_loads;
    Kokkos::View<double*> vector;

    KOKKOS_FUNCTION
    void operator()(size_t i_node) const {
        const auto this_node_freedom_signature = node_freedom_allocation_table(i_node);
        const auto dof_index = node_freedom_map_table(i_node);
        if (this_node_freedom_signature == FreedomSignature::NoComponents) {
            return;
        } else if (this_node_freedom_signature == FreedomSignature::AllComponents) {
            vector(dof_index + 0) -= node_loads(i_node, 0);
            vector(dof_index + 1) -= node_loads(i_node, 1);
            vector(dof_index + 2) -= node_loads(i_node, 2);
            vector(dof_index + 3) -= node_loads(i_node, 3);
            vector(dof_index + 4) -= node_loads(i_node, 4);
            vector(dof_index + 5) -= node_loads(i_node, 5);
        } else if (this_node_freedom_signature == FreedomSignature::JustPosition) {
            vector(dof_index + 0) -= node_loads(i_node, 0);
            vector(dof_index + 1) -= node_loads(i_node, 1);
            vector(dof_index + 2) -= node_loads(i_node, 2);
        } else if (this_node_freedom_signature == FreedomSignature::JustRotation) {
            vector(dof_index + 0) -= node_loads(i_node, 3);
            vector(dof_index + 1) -= node_loads(i_node, 4);
            vector(dof_index + 2) -= node_loads(i_node, 5);
        }
    }
};

}  // namespace openturbine
