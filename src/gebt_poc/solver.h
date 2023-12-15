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
    Kokkos::View<double*> nodal_values, std::vector<double> interpolation_function,
    Kokkos::View<double*> interpolated_values
);

/// Calculates the interpolated derivative values for a nodal quantity (e.g. displacement or position
/// vector) at a given quadrature point
void InterpolateNodalValueDerivatives(
    Kokkos::View<double*> nodal_values, std::vector<double> interpolation_function,
    const double jacobian, Kokkos::View<double*> interpolated_values
);

/// Calculates the curvature from generalized coordinates and their derivatives
void CalculateCurvature(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> gen_coords_derivative,
    const Kokkos::View<double*> curvature
);

/// Calculates the given sectional stiffness matrix in inertial basis based on the given
/// rotation matrices
void CalculateSectionalStiffness(
    const StiffnessMatrix& stiffness, Kokkos::View<double**> rotation_0,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness
);

/// Calculates the elastic forces based on the sectional strain, derivative of the position
/// vector and the generalized coordinates, and the sectional stiffness matrix
void CalculateElasticForces(
    const Kokkos::View<double*> strain, Kokkos::View<double**> rotation,
    const Kokkos::View<double*> pos_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness,
    Kokkos::View<double[kNumberOfLieGroupComponents]> elastic_forces_fc,
    Kokkos::View<double[kNumberOfLieGroupComponents]> elastic_forces_fd
);

/// Calculates the static residual vector for a beam element
void CalculateStaticResidual(
    const Kokkos::View<double*> position_vectors, const Kokkos::View<double*> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature, Kokkos::View<double*> residual
);

/// Calculates the given sectional mass matrix in inertial basis based on the given
/// rotation matrices
void CalculateSectionalMassMatrix(
    const MassMatrix& mass_matrix, Kokkos::View<double**> rotation_0,
    Kokkos::View<double[kNumberOfVectorComponents][kNumberOfVectorComponents]> rotation,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_mass_matrix
);

/// Calculates the inertial forces based on the sectional mass matrix, velocity, and acceleration
void CalculateInertialForces(
    Kokkos::View<double*> velocity, Kokkos::View<double*> acceleration,
    const MassMatrix& sectional_mass_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents]> inertial_forces_fc
);

void CalculateIterationMatrixComponents(
    const Kokkos::View<double*> elastic_force_fc, const Kokkos::View<double*> pos_vector_derivatives,
    const Kokkos::View<double*> gen_coords_derivatives,
    const Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]>
        sectional_stiffness,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> O_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> P_matrix,
    Kokkos::View<double[kNumberOfLieGroupComponents][kNumberOfLieGroupComponents]> Q_matrix
);

/// Calculates the static iteration matrix for a beam element
void CalculateStaticIterationMatrix(
    const Kokkos::View<double*> position_vectors, const Kokkos::View<double*> gen_coords,
    const StiffnessMatrix& stiffness, const Quadrature& quadrature,
    Kokkos::View<double**> iteration_matrix
);

/// Calculates the constraint residual vector for a beam element
void ConstraintsResidualVector(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> position_vector,
    const Kokkos::View<double*> constraints_residual
);

/// Calculates the constraint gradient matrix for a beam element
void ConstraintsGradientMatrix(
    const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> position_vector,
    Kokkos::View<double**> constraints_gradient_matrix
);

}  // namespace openturbine::gebt_poc
