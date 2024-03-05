#pragma once

#include "src/gebt_poc/element.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

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
