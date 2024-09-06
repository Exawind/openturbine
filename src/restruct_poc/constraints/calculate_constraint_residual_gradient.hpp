#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_cylindrical_constraint.hpp"
#include "calculate_fixed_bc_constraint.hpp"
#include "calculate_prescribed_bc_constraint.hpp"
#include "calculate_rigid_constraint.hpp"
#include "calculate_rotation_control_constraint.hpp"
#include "constraints.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [3][3]>::const_type axes;
    View_N::const_type control;
    View_Nx7::const_type constraint_u;
    View_Nx7::const_type node_u;
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6][6]> base_gradient_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        if (type(i_constraint) == ConstraintType::kFixedBC) {
            CalculateFixedBCConstraint{target_node_index,    X0_,    control,
                                       constraint_u,         node_u, residual_terms,
                                       target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kPrescribedBC) {
            CalculatePrescribedBCConstraint{target_node_index,    X0_,    control,
                                            constraint_u,         node_u, residual_terms,
                                            target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kRigid) {
            CalculateRigidConstraint{
                base_node_index, target_node_index,   X0_,
                control,         constraint_u,        node_u,
                residual_terms,  base_gradient_terms, target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kCylindrical) {
            CalculateCylindricalConstraint{base_node_index,
                                           target_node_index,
                                           X0_,
                                           axes,
                                           control,
                                           constraint_u,
                                           node_u,
                                           residual_terms,
                                           base_gradient_terms,
                                           target_gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kRotationControl) {
            CalculateRotationControlConstraint{base_node_index,
                                               target_node_index,
                                               X0_,
                                               axes,
                                               control,
                                               constraint_u,
                                               node_u,
                                               residual_terms,
                                               base_gradient_terms,
                                               target_gradient_terms}(i_constraint);
        }
    }
};

}  // namespace openturbine
