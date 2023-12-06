#pragma once

#include "src/gen_alpha_poc/linearization_parameters.h"
#include "src/gen_alpha_poc/state.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gen_alpha_solver {

/*!
 * Calculates the residual vector and iteration matrix for the heavy top problem from Brüls and
 * Cardona (2010) "On the use of Lie group time integrators in multibody dynamics," 2010, Journal
 * of Computational and Nonlinear Dynamics, Vol 5.
 * Ref: https://doi.org/10.1115/1.4001370
 */
class HeavyTopLinearizationParameters : public LinearizationParameters {
public:
    static constexpr size_t kNumberOfLieGroupComponents = 7;
    static constexpr size_t kNumberOfLieAlgebraComponents = 6;
    static constexpr size_t kNumberOfConstraints = 3;

    HeavyTopLinearizationParameters();

    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> velocity,
        const Kokkos::View<double*> acceleration, const Kokkos::View<double*> lagrange_multipliers
    ) override;

    /// Calculates the iteration matrix for the heavy top problem
    virtual Kokkos::View<double**> IterationMatrix(
        const double& h, const double& beta_prime, const double& gamma_prime,
        const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> delta_gen_coords,
        const Kokkos::View<double*> velocity, const Kokkos::View<double*> acceleration,
        const Kokkos::View<double*> lagrange_mults
    ) override;

    /// Calculates the generalized coordinates residual vector for the heavy top problem
    Kokkos::View<double*> GeneralizedCoordinatesResidualVector(
        const Kokkos::View<double**> mass_matrix, const Kokkos::View<double**> rotation_matrix,
        const Kokkos::View<double*> acceleration_vector,
        const Kokkos::View<double*> gen_forces_vector,
        const Kokkos::View<double*> lagrange_multipliers,
        const Kokkos::View<double*> reference_position_vector
    );

    /// Calculates the constraint residual vector for the heavy top problem
    Kokkos::View<double*> ConstraintsResidualVector(
        const Kokkos::View<double**> rotation_matrix, const Kokkos::View<double*> position_vector,
        const Kokkos::View<double*> reference_position_vector
    );

    /// Calculates the constraint gradient matrix for the heavy top problem
    Kokkos::View<double**> ConstraintsGradientMatrix(
        const Kokkos::View<double**> rotation_matrix,
        const Kokkos::View<double*> reference_position_vector
    );

    /// Calculates the tangent damping matrix for the heavy top problem
    Kokkos::View<double**> TangentDampingMatrix(
        const Kokkos::View<double*> angular_velocity_vector,
        const Kokkos::View<double**> inertia_matrix
    );

    /// Calculates the tangent stiffness matrix for the heavy top problem
    Kokkos::View<double**> TangentStiffnessMatrix(
        const Kokkos::View<double**> rotation_matrix,
        const Kokkos::View<double*> lagrange_multipliers,
        const Kokkos::View<double*> reference_position_vector
    );

    Kokkos::View<double**> TangentOperator(const Kokkos::View<double*> psi);

private:
    double mass_;
    Vector gravity_;
    Vector principal_moment_of_inertia_;
    MassMatrix mass_matrix_;

    Kokkos::View<double**> CalculateRotationMatrix(const Kokkos::View<double*>);
    Kokkos::View<double*> CalculateForces(MassMatrix, const Kokkos::View<double*>);
};

}  // namespace openturbine::gen_alpha_solver
