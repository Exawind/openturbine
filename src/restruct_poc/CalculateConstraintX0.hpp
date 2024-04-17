#pragma once

#include <Kokkos_Core.hpp>

#include "Constraints.hpp"
#include "types.hpp"

namespace openturbine {

struct CalculateConstraintX0 {
    Kokkos::View<Constraints::NodeIndices*>::const_type node_indices;
    View_Nx7::const_type node_x0;
    View_Nx3 constraint_X0;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto i_node1 = node_indices(i_constraint).base_node_index;
        auto i_node2 = node_indices(i_constraint).constrained_node_index;

        if (i_node1 == -1) {
            constraint_X0(i_constraint, 0) = node_x0(i_node2, 0);
            constraint_X0(i_constraint, 1) = node_x0(i_node2, 1);
            constraint_X0(i_constraint, 2) = node_x0(i_node2, 2);
        } else {
            constraint_X0(i_constraint, 0) = node_x0(i_node2, 0) - node_x0(i_node1, 0);
            constraint_X0(i_constraint, 1) = node_x0(i_node2, 1) - node_x0(i_node1, 1);
            constraint_X0(i_constraint, 2) = node_x0(i_node2, 2) - node_x0(i_node1, 2);
        }
    }
};

}
