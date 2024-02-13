#pragma once

#include "src/gebt_poc/element.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/state.h"
#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

constexpr std::size_t kNumberOfLieAlgebraComponents = 7;
constexpr std::size_t kNumberOfLieGroupComponents = 6;
constexpr std::size_t kNumberOfVectorComponents = 3;

/// Calculates the interpolated values for a nodal quantity (e.g. displacement or position vector)
/// at a given quadrature point and normalizes the rotation quaternion
void InterpolateNodalValues(
    Kokkos::View<const double**> nodal_values, std::vector<double> interpolation_function,
    Kokkos::View<double*> interpolated_value
);

/// Calculates the interpolated derivative values for a nodal quantity (e.g. displacement
/// or position vector) at a given quadrature point
void InterpolateNodalValueDerivatives(
    Kokkos::View<const double**> nodal_values, std::vector<double> interpolation_function,
    const double jacobian, Kokkos::View<double*> interpolated_values
);

/// Calculates the curvature from generalized coordinates and their derivatives
void NodalCurvature(
    const Kokkos::View<double[kNumberOfLieAlgebraComponents]> gen_coords,
    const Kokkos::View<double[kNumberOfLieAlgebraComponents]> gen_coords_derivative,
    const Kokkos::View<double[kNumberOfVectorComponents]> curvature
);

/// Transforms the given 6 x 6 stiffness matrix in material basis -> inertial basis based on the
/// given rotation matrices
void SectionalStiffness(
    const StiffnessMatrix& stiffness,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation_0,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness
);

/// Calculates the elastic forces based on the sectional strain, derivative of the position
/// vector and the generalized coordinates, and the sectional stiffness matrix
void NodalElasticForces(
    const Kokkos::View<double[kNumberOfLieGroupComponents]> sectional_strain,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation,
    const Kokkos::View<double[kNumberOfLieAlgebraComponents]> pos_vector_derivatives,
    const Kokkos::View<double[kNumberOfLieAlgebraComponents]> gen_coords_derivatives,
    const Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness,
    Kokkos::View<double[kNumberOfLieGroupComponents]> elastic_forces_fc,
    Kokkos::View<double[kNumberOfLieGroupComponents]> elastic_forces_fd
);

/// Calculates the static residual vector for a beam element
void ElementalStaticForcesResidual(
    Kokkos::View<const double**> position_vectors, Kokkos::View<const double**> gen_coords, const StiffnessMatrix& stiffness, const Quadrature& quadrature, Kokkos::View<double*> residual
);

/// Transforms the provided 6 x 6 mass matrix in material/current configuration -> inertial basis
/// based on the given rotation matrices
void SectionalMassMatrix(
    const MassMatrix& mass_matrix,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation_0,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_mass_matrix
);

/// Calculates the inertial forces based on the sectional mass matrix, velocity, and acceleration
void NodalInertialForces(
    Kokkos::View<double[kNumberOfLieGroupComponents]> velocity,
    Kokkos::View<double[kNumberOfLieGroupComponents]> acceleration,
    const MassMatrix& sectional_mass_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents]> inertial_forces_fc
);

/// Calculates the dynamic residual vector for a beam element
void ElementalInertialForcesResidual(
    Kokkos::View<const double**> position_vectors, Kokkos::View<const double**> gen_coords,
    Kokkos::View<const double**> velocity, const Kokkos::View<const double**> acceleration,
    const MassMatrix& mass_matrix, const Quadrature& quadrature, Kokkos::View<double*> residual
);

void NodalStaticStiffnessMatrixComponents(
    const Kokkos::View<double[kNumberOfLieGroupComponents]> elastic_force_fc,
    const Kokkos::View<double[kNumberOfLieAlgebraComponents]> pos_vector_derivatives,
    const Kokkos::View<double[kNumberOfLieAlgebraComponents]> gen_coords_derivatives,
    const Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> O_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> P_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> Q_matrix
);

/// Calculates the static iteration matrix for a beam element
void ElementalStaticStiffnessMatrix(
    Kokkos::View<const double**> position_vectors, Kokkos::View<const double**> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature,
    Kokkos::View<double**> stiffness_matrix
);

void NodalGyroscopicMatrix(
    Kokkos::View<double[kNumberOfLieGroupComponents]> velocity,
    const MassMatrix& sectional_mass_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> gyroscopic_matrix
);

void NodalDynamicStiffnessMatrix(
    Kokkos::View<double[kNumberOfLieGroupComponents]> velocity,
    Kokkos::View<double[kNumberOfLieGroupComponents]> acceleration,
    const MassMatrix& sectional_mass_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> stiffness_matrix
);

void ElementalInertialMatrices(
    Kokkos::View<const double**> position_vectors, Kokkos::View<const double**> gen_coords,
    Kokkos::View<const double**> velocity, Kokkos::View<const double**> acceleration,
    const MassMatrix& mass_matrix, const Quadrature& quadrature,
    Kokkos::View<double**> element_mass_matrix, Kokkos::View<double**> element_gyroscopic_matrix,
    Kokkos::View<double**> element_dynamic_stiffness_matrix
);

/// Calculates the constraint residual vector for a beam element
void ElementalConstraintForcesResidual(
    Kokkos::View<const double**> gen_coords, Kokkos::View<double*> constraints_residual
);

/// Calculates the constraint gradient matrix for a beam element
void ElementalConstraintForcesGradientMatrix(
    Kokkos::View<const double**> gen_coords, Kokkos::View<const double**> position_vector,
    Kokkos::View<double**> constraints_gradient_matrix
);

}  // namespace openturbine::gebt_poc
