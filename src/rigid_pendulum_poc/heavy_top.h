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
    HeavyTopLinearizationParameters(){};

    virtual HostView1D ResidualVector(
        const HostView1D, const HostView1D, const HostView1D, const HostView1D
    ) override;

    virtual HostView2D IterationMatrix(
        const double&, const double&, const HostView1D, const HostView1D, const HostView1D,
        const double& = 1., const HostView1D = create_vector({1., 1., 1.})
    ) override;

    /// Calculates the generalized coordinates residual vector for the heavy top problem
    HostView1D GeneralizedCoordinatesResidualVector(
        const HostView2D, const HostView2D, const HostView1D, const HostView1D, const HostView1D,
        const HostView1D reference_position_vector
    );

    /// Calculates the constraint residual vector for the heavy top problem
    HostView1D ConstraintsResidualVector(
        const HostView2D, const HostView1D, const HostView1D reference_position_vector
    );

    /// Calculates the constraint gradient matrix for the heavy top problem
    HostView2D ConstraintsGradientMatrix(
        const HostView2D, const HostView1D reference_position_vector
    );

    /// Calculates the tangent damping matrix for the heavy top problem
    HostView2D TangentDampingMatrix(const HostView1D, const HostView2D);

    /// Calculates the tangent stiffness matrix for the heavy top problem
    HostView2D TangentStiffnessMatrix(
        const HostView2D, const HostView1D, const HostView1D reference_position_vector
    );

    HostView2D TangentOperator(const HostView1D psi);

private:
};

}  // namespace openturbine::rigid_pendulum
