#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/solver/calculate_revolute_joint_force.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct AssembleConstraintForce {
    Kokkos::View<Constraints::DeviceData*>::const_type data;
    View_N::const_type control;
    View_Nx7::const_type node_u;
    View_N R_system;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        const auto& cd = data(i_constraint);
        switch (cd.type) {
            case ConstraintType::kRevoluteJoint: {
                // Applies the torque from a revolute joint constraint to the system residual
                CalculateRevoluteJointForce{data, control, node_u, R_system}(i_constraint);
            } break;
            default: {
                // Do nothing
            } break;
        }
    }
};

}  // namespace openturbine
