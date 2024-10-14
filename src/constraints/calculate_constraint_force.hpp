#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_revolute_joint_force.hpp"
#include "constraint_type.hpp"

#include "src/types.hpp"

namespace openturbine {

struct CalculateConstraintForce {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3][3]>::const_type axes;
    Kokkos::View<double* [7]>::const_type inputs;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]> system_residual_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        switch (type(i_constraint)) {
            case ConstraintType::kRevoluteJoint: {
                // Applies the torque from a revolute joint constraint to the system residual
                CalculateRevoluteJointForce{
                    target_node_index, axes, inputs, node_u, system_residual_terms
                }(i_constraint);
            } break;
            default: {
                // Do nothing
            } break;
        }
    }
};

}  // namespace openturbine
