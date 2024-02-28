#pragma once

#include "src/gebt_poc/element.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

/// Calculates the interpolated values for a nodal quantity (e.g. displacement or position vector)
/// at a given quadrature point and normalizes the rotation quaternion
void InterpolateNodalValues(
    View1D::const_type nodal_values, std::vector<double> interpolation_function,
    View1D interpolated_values, std::size_t n_components = LieGroupComponents
);

/// Calculates the interpolated derivative values for a nodal quantity (e.g. displacement
/// or position vector) at a given quadrature point
void InterpolateNodalValueDerivatives(
    View1D::const_type nodal_values, std::vector<double> interpolation_function, double jacobian,
    View1D interpolated_values
);

/// Calculates the curvature from generalized coordinates and their derivatives
void NodalCurvature(
    View1D_LieGroup::const_type gen_coords, View1D_LieGroup::const_type gen_coords_derivative,
    View1D_Vector curvature
);

/// Transforms the given 6 x 6 stiffness matrix in material basis -> inertial basis based on the
/// given rotation matrices
void SectionalStiffness(
    const StiffnessMatrix& stiffness, View2D_3x3::const_type rotation_0,
    View2D_3x3::const_type rotation, View2D_6x6 sectional_stiffness
);

/// Calculates the elastic forces based on the sectional strain, derivative of the position
/// vector and the generalized coordinates, and the sectional stiffness matrix
void NodalElasticForces(
    View1D_LieAlgebra::const_type sectional_strain, View2D_3x3::const_type rotation,
    View1D_LieGroup::const_type pos_vector_derivatives,
    View1D_LieGroup::const_type gen_coords_derivatives, View2D_6x6::const_type sectional_stiffness,
    View1D_LieAlgebra elastic_forces_fc, View1D_LieAlgebra elastic_forces_fd
);

/// Calculates the static residual vector for a beam element
void ElementalStaticForcesResidual(
    View1D::const_type position_vectors, View1D::const_type gen_coords,
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

/// Calculates the constraint residual vector for a beam element
void ElementalConstraintForcesResidual(View1D::const_type gen_coords, View1D constraints_residual);

/// Calculates the constraint gradient matrix for a beam element
void ElementalConstraintForcesGradientMatrix(
    View1D::const_type gen_coords, View1D::const_type position_vector,
    View2D constraints_gradient_matrix
);

}  // namespace openturbine::gebt_poc
