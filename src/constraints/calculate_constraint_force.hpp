#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_revolute_joint_force.hpp"
#include "constraint_type.hpp"

#include "src/types.hpp"

namespace openturbine {

struct CalculateConstraintForce {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type target_node_index;
    View_Nx3x3::const_type axes;
    View_Nx7::const_type inputs;
    View_Nx7::const_type node_u;
    View_Nx6 system_residual_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        if (type(i_constraint) == ConstraintType::kRevoluteJoint) {
            // Applies the torque from a revolute joint constraint to the system residual
            CalculateRevoluteJointForce{i_constraint, target_node_index,    axes, inputs,
                                        node_u,       system_residual_terms}();
            return;
        }
    }
};

}  // namespace openturbine
