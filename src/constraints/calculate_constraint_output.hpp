#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_revolute_joint_output.hpp"
#include "constraint_type.hpp"

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
        if (type(i_constraint) == ConstraintType::kRevoluteJoint) {
            CalculateRevoluteJointOutput{i_constraint, target_node_index, axes,       node_x0,
                                         node_u,       node_udot,         node_uddot, outputs}();
            return;
        }
    }
};

}  // namespace openturbine
