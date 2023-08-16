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

    virtual HostView1D ResidualVector(
        const HostView1D gen_coords, const HostView1D velocity, const HostView1D acceleration,
        const HostView1D lagrange_multipliers
    ) override;

    /// Calculates the iteration matrix for the heavy top problem
    virtual HostView2D IterationMatrix(
        const double& h, const double& beta_prime, const double& gamma_prime,
        const HostView1D gen_coords, const HostView1D delta_gen_coords, const HostView1D velocity,
        const HostView1D acceleration, const HostView1D lagrange_mults
    ) override;

    /// Calculates the generalized coordinates residual vector for the heavy top problem
    HostView1D GeneralizedCoordinatesResidualVector(
        const HostView2D mass_matrix, const HostView2D rotation_matrix,
        const HostView1D acceleration_vector, const HostView1D gen_forces_vector,
        const HostView1D lagrange_multipliers, const HostView1D reference_position_vector
    );

    /// Calculates the constraint residual vector for the heavy top problem
    HostView1D ConstraintsResidualVector(
        const HostView2D rotation_matrix, const HostView1D position_vector,
        const HostView1D reference_position_vector
    );

    /// Calculates the constraint gradient matrix for the heavy top problem
    HostView2D ConstraintsGradientMatrix(
        const HostView2D rotation_matrix, const HostView1D reference_position_vector
    );

    /// Calculates the tangent damping matrix for the heavy top problem
    HostView2D TangentDampingMatrix(
        const HostView1D angular_velocity_vector, const HostView2D inertia_matrix
    );

    /// Calculates the tangent stiffness matrix for the heavy top problem
    HostView2D TangentStiffnessMatrix(
        const HostView2D rotation_matrix, const HostView1D lagrange_multipliers,
        const HostView1D reference_position_vector
    );

    HostView2D TangentOperator(const HostView1D psi);

private:
    double mass_;
    Vector gravity_;
    Vector principal_moment_of_inertia_;
    MassMatrix mass_matrix_;

    HostView2D CalculateRotationMatrix(const HostView1D);
    HostView1D CalculateForces(MassMatrix, const HostView1D);
};

}  // namespace openturbine::gen_alpha_solver
