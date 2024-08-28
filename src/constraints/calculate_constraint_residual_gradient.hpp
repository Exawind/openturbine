#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_cylindrical_constraint.hpp"
#include "calculate_fixed_bc_constraint.hpp"
#include "calculate_prescribed_bc_constraint.hpp"
#include "calculate_rigid_constraint.hpp"
#include "calculate_rotation_control_constraint.hpp"
#include "constraints.hpp"

#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t* [2]>::const_type node_index;
    Kokkos::View<size_t* [2]>::const_type row_range;
    Kokkos::View<size_t* [2][2]>::const_type node_col_range;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [3][3]>::const_type axis;
    Kokkos::View<double*>::const_type control;
    Kokkos::View<double* [7]>::const_type constraint_u;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double*> Phi_;
    Kokkos::View<double* [6][12]> gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        if (type(i_constraint) == ConstraintType::kFixedBC) {
            CalculateFixedBCConstraint{node_index, row_range, node_col_range,
                                       X0_,        control,   constraint_u,
                                       node_u,     Phi_,      gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kPrescribedBC) {
            CalculatePrescribedBCConstraint{node_index, row_range, node_col_range,
                                            X0_,        control,   constraint_u,
                                            node_u,     Phi_,      gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kRigid) {
            CalculateRigidConstraint{
                node_index,   row_range, node_col_range, X0_,           control,
                constraint_u, node_u,    Phi_,           gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kCylindrical) {
            CalculateCylindricalConstraint{node_index, row_range,     node_col_range, X0_,
                                           axis,       control,       constraint_u,   node_u,
                                           Phi_,       gradient_terms}(i_constraint);
        } else if (type(i_constraint) == ConstraintType::kRotationControl) {
            CalculateRotationControlConstraint{node_index, row_range,     node_col_range, X0_,
                                               axis,       control,       constraint_u,   node_u,
                                               Phi_,       gradient_terms}(i_constraint);
        }
    }
};

}  // namespace openturbine
