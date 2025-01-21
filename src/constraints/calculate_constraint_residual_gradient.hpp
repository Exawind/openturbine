#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_fixed_bc_constraint.hpp"
#include "calculate_prescribed_bc_constraint.hpp"
#include "calculate_revolute_joint_constraint.hpp"
#include "calculate_rigid_joint_constraint.hpp"
#include "calculate_rotation_control_constraint.hpp"
#include "constraints.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    View_Nx3::const_type X0_;
    View_Nx3x3::const_type axes;
    View_Nx7::const_type constraint_inputs;
    View_Nx7::const_type node_u;
    View_Nx6 residual_terms;
    View_Nx6x6 base_gradient_terms;
    View_Nx6x6 target_gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto constraint_type = type(i_constraint);
        if (constraint_type == ConstraintType::kFixedBC ||
            constraint_type == ConstraintType::kFixedBC3DOFs) {
            CalculateFixedBCConstraint{
                i_constraint, target_node_col_range, target_node_index,     X0_, constraint_inputs,
                node_u,       residual_terms,        target_gradient_terms,
            }();
            return;
        };
        if (constraint_type == ConstraintType::kPrescribedBC ||
            constraint_type == ConstraintType::kPrescribedBC3DOFs) {
            CalculatePrescribedBCConstraint{
                i_constraint, target_node_col_range, target_node_index,     X0_, constraint_inputs,
                node_u,       residual_terms,        target_gradient_terms,
            }();
            return;
        };
        if (constraint_type == ConstraintType::kRigidJoint ||
            constraint_type == ConstraintType::kRigidJoint6DOFsTo3DOFs) {
            CalculateRigidJointConstraint{
                i_constraint,
                target_node_col_range,
                base_node_index,
                target_node_index,
                X0_,
                constraint_inputs,
                node_u,
                residual_terms,
                base_gradient_terms,
                target_gradient_terms,
            }();
            return;
        };
        if (constraint_type == ConstraintType::kRevoluteJoint) {
            CalculateRevoluteJointConstraint{
                i_constraint,
                base_node_index,
                target_node_index,
                X0_,
                axes,
                constraint_inputs,
                node_u,
                residual_terms,
                base_gradient_terms,
                target_gradient_terms,
            }();
            return;
        };
        if (constraint_type == ConstraintType::kRotationControl) {
            CalculateRotationControlConstraint{
                i_constraint,
                base_node_index,
                target_node_index,
                X0_,
                axes,
                constraint_inputs,
                node_u,
                residual_terms,
                base_gradient_terms,
                target_gradient_terms,
            }();
            return;
        }
    };
};

}  // namespace openturbine
