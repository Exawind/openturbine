#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_fixed_bc_constraint.hpp"
#include "calculate_prescribed_bc_constraint.hpp"
#include "calculate_revolute_joint_constraint.hpp"
#include "calculate_rigid_joint_constraint.hpp"
#include "calculate_rotation_control_constraint.hpp"
#include "constraints.hpp"

#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [3][3]>::const_type axes;
    Kokkos::View<double* [7]>::const_type constraint_inputs;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6][6]> base_gradient_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        if (type(i_constraint) == ConstraintType::kFixedBC) {
            CalculateFixedBCConstraint{target_node_index, X0_,
                                       constraint_inputs, node_u,
                                       residual_terms,    target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kPrescribedBC) {
            CalculatePrescribedBCConstraint{target_node_index, X0_,
                                            constraint_inputs, node_u,
                                            residual_terms,    target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kRigidJoint) {
            CalculateRigidJointConstraint{base_node_index,
                                          target_node_index,
                                          X0_,
                                          constraint_inputs,
                                          node_u,
                                          residual_terms,
                                          base_gradient_terms,
                                          target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kRevoluteJoint) {
            CalculateRevoluteJointConstraint{base_node_index,
                                             target_node_index,
                                             X0_,
                                             axes,
                                             constraint_inputs,
                                             node_u,
                                             residual_terms,
                                             base_gradient_terms,
                                             target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kRotationControl) {
            CalculateRotationControlConstraint{base_node_index,
                                               target_node_index,
                                               X0_,
                                               axes,
                                               constraint_inputs,
                                               node_u,
                                               residual_terms,
                                               base_gradient_terms,
                                               target_gradient_terms}(i_constraint);
        }
    }
};

}  // namespace openturbine
