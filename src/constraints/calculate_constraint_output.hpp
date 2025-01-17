#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_revolute_joint_output.hpp"
#include "constraint_type.hpp"

namespace openturbine {

struct CalculateConstraintOutput {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type target_node_index;
    View_Nx3x3::const_type axes;
    View_Nx7::const_type node_x0;     // Initial position
    View_Nx7::const_type node_u;      // Displacement
    View_Nx6::const_type node_udot;   // Velocity
    View_Nx6::const_type node_uddot;  // Acceleration
    View_Nx3 outputs;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        if (type(i_constraint) == ConstraintType::kRevoluteJoint) {
            CalculateRevoluteJointOutput{i_constraint, target_node_index, axes,       node_x0,
                                         node_u,       node_udot,         node_uddot, outputs}();
            return;
        }
    }
};

}  // namespace openturbine
