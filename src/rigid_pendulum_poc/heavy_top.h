#pragma once

#include "src/rigid_pendulum_poc/linearization_parameters.h"
#include "src/rigid_pendulum_poc/state.h"
#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

/*! Calculates the residual vector and iteration matrix for the heavy top problem from Brüls and
 * Cardona (2010) "On the use of Lie group time integrators in multibody dynamics," 2010, Journal
 * of Computational and Nonlinear Dynamics, Vol 5.
 * Ref: https://doi.org/10.1115/1.4001370
 */
class HeavyTopLinearizationParameters : public LinearizationParameters {
public:
    HeavyTopLinearizationParameters();

    virtual Kokkos::View<double*>
    ResidualVector(const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>)
        override;

    virtual Kokkos::View<double**>
    IterationMatrix(const double&, const double&, const double&, const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>)
        override;

    /// Calculates the generalized coordinates residual vector for the heavy top problem
    Kokkos::View<double*> GeneralizedCoordinatesResidualVector(
        const Kokkos::View<double**>, const Kokkos::View<double**>, const Kokkos::View<double*>,
        const Kokkos::View<double*>, const Kokkos::View<double*>,
        const Kokkos::View<double*> reference_position_vector
    );

    /// Calculates the constraint residual vector for the heavy top problem
    Kokkos::View<double*> ConstraintsResidualVector(
        const Kokkos::View<double**>, const Kokkos::View<double*>,
        const Kokkos::View<double*> reference_position_vector
    );

    /// Calculates the constraint gradient matrix for the heavy top problem
    Kokkos::View<double**> ConstraintsGradientMatrix(
        const Kokkos::View<double**>, const Kokkos::View<double*> reference_position_vector
    );

    /// Calculates the tangent damping matrix for the heavy top problem
    Kokkos::View<double**>
    TangentDampingMatrix(const Kokkos::View<double*>, const Kokkos::View<double**>);

    /// Calculates the tangent stiffness matrix for the heavy top problem
    Kokkos::View<double**> TangentStiffnessMatrix(
        const Kokkos::View<double**>, const Kokkos::View<double*>,
        const Kokkos::View<double*> reference_position_vector
    );

    Kokkos::View<double**> TangentOperator(const Kokkos::View<double*> psi);

private:
    MassMatrix mass_matrix_;

    Kokkos::View<double**> CalculateRotationMatrix(const Kokkos::View<double*>);
    Kokkos::View<double*> CalculateForces(MassMatrix, const Kokkos::View<double*>);
};

}  // namespace openturbine::rigid_pendulum
