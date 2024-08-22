#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_fixed_bc_constraint.hpp"
#include "calculate_prescribed_bc_constraint.hpp"
#include "calculate_revolute_joint_constraint.hpp"
#include "calculate_rigid_joint_constraint.hpp"
#include "calculate_rotation_control_constraint.hpp"
#include "constraints.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<Constraints::DeviceData*>::const_type data;
    View_N::const_type control;
    View_Nx7::const_type constraint_u;
    View_Nx7::const_type node_u;
    View_N Phi_;
    Kokkos::View<double* [6][12]> gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        const auto& cd = data(i_constraint);
        if (cd.type == ConstraintType::kFixedBC) {
            CalculateFixedBCConstraint{
                data, control, constraint_u, node_u, Phi_, gradient_terms}(i_constraint);
        } else if (cd.type == ConstraintType::kPrescribedBC) {
            CalculatePrescribedBCConstraint{data,   control, constraint_u,
                                            node_u, Phi_,    gradient_terms}(i_constraint);
        } else if (cd.type == ConstraintType::kRigidJoint) {
            CalculateRigidJointConstraint{
                data, control, constraint_u, node_u, Phi_, gradient_terms}(i_constraint);
        } else if (cd.type == ConstraintType::kRevoluteJoint) {
            CalculateRevoluteJointConstraint{data,   control, constraint_u,
                                             node_u, Phi_,    gradient_terms}(i_constraint);
        } else if (cd.type == ConstraintType::kRotationControl) {
            CalculateRotationControlConstraint{data,   control, constraint_u,
                                               node_u, Phi_,    gradient_terms}(i_constraint);
        }
    }
};

}  // namespace openturbine
