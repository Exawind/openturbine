#pragma once

#include "src/gebt_poc/element.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {
/// Calculates the static residual vector for a beam element
void ElementalStaticForcesResidual(
    View1D::const_type position_vectors, View1D::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View1D residual
);

void ElementalStaticForcesResidual(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View1D residual
);

/// Transforms the provided 6 x 6 mass matrix in material/current configuration -> inertial basis
/// based on the given rotation matrices
void SectionalMassMatrix(
    const MassMatrix& mass_matrix, View2D_3x3 rotation_0, View2D_3x3 rotation,
    View2D_6x6 sectional_mass_matrix
);

/// Calculates the inertial forces based on the sectional mass matrix, velocity, and acceleration
void NodalInertialForces(
    View1D_LieAlgebra::const_type velocity, View1D_LieAlgebra::const_type acceleration,
    const MassMatrix& sectional_mass_matrix, View1D_LieAlgebra inertial_forces_fc
);

/// Calculates the dynamic residual vector for a beam element
void ElementalInertialForcesResidual(
    View1D::const_type position_vectors, View1D::const_type gen_coords, View1D::const_type velocity,
    View1D::const_type acceleration, const MassMatrix& mass_matrix, const Quadrature& quadrature,
    View1D residual
);
void ElementalInertialForcesResidual(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type velocity, LieAlgebraFieldView::const_type acceleration,
    const MassMatrix& mass_matrix, const Quadrature& quadrature, View1D residual
);

void NodalStaticStiffnessMatrixComponents(
    View1D_LieAlgebra::const_type elastic_force_fc,
    View1D_LieGroup::const_type pos_vector_derivatives,
    View1D_LieGroup::const_type gen_coords_derivatives, View2D_6x6::const_type sectional_stiffness,
    View2D_6x6 O_matrix, View2D_6x6 P_matrix, View2D_6x6 Q_matrix
);

/// Calculates the static stiffness matrix for a beam element
void ElementalStaticStiffnessMatrix(
    View1D::const_type position_vectors, View1D::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View2D stiffness_matrix
);
void ElementalStaticStiffnessMatrix(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, View2D stiffness_matrix
);

void NodalGyroscopicMatrix(
    View1D_LieAlgebra::const_type velocity, const MassMatrix& sectional_mass_matrix,
    View2D_6x6 gyroscopic_matrix
);

void NodalDynamicStiffnessMatrix(
    View1D_LieAlgebra::const_type velocity, View1D_LieAlgebra::const_type acceleration,
    const MassMatrix& sectional_mass_matrix, View2D_6x6 stiffness_matrix
);

void ElementalInertialMatrices(
    View1D::const_type position_vectors, View1D::const_type gen_coords, View1D::const_type velocity,
    View1D::const_type acceleration, const MassMatrix& mass_matrix, const Quadrature& quadrature,
    View2D element_mass_matrix, View2D element_gyroscopic_matrix,
    View2D element_dynamic_stiffness_matrix
);
void ElementalInertialMatrices(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type velocity, LieAlgebraFieldView::const_type acceleration,
    const MassMatrix& mass_matrix, const Quadrature& quadrature, View2D element_mass_matrix,
    View2D element_gyroscopic_matrix, View2D element_dynamic_stiffness_matrix
);

/// Calculates the constraint residual vector for a beam element
void ElementalConstraintForcesResidual(View1D::const_type gen_coords, View1D constraints_residual);
void ElementalConstraintForcesResidual(
    LieGroupFieldView::const_type gen_coords, View1D constraints_residual
);

/// Calculates the constraint gradient matrix for a beam element
void ElementalConstraintForcesGradientMatrix(
    View1D::const_type gen_coords, View1D::const_type position_vector,
    View2D constraints_gradient_matrix
);

}  // namespace openturbine::gebt_poc
