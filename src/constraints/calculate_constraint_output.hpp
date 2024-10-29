#pragma once

#include <Kokkos_Core.hpp>

#include "constraint_type.hpp"

#include "calculate_revolute_joint_output.hpp"

namespace openturbine {

struct CalculateConstraintOutput {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3][3]>::const_type axes;
    Kokkos::View<double* [7]>::const_type node_x0;     // Initial position
    Kokkos::View<double* [7]>::const_type node_u;      // Displacement
    Kokkos::View<double* [6]>::const_type node_udot;   // Velocity
    Kokkos::View<double* [6]>::const_type node_uddot;  // Acceleration
    Kokkos::View<double* [3]> outputs;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        switch (type(i_constraint)) {
            case ConstraintType::kRevoluteJoint: {
                CalculateRevoluteJointOutput{target_node_index, axes, node_x0, node_u, node_udot, node_uddot, outputs}(i_constraint);
            } break;
            default: {
                // Do nothing
            } break;
        }
    }
};

}  // namespace openturbine
