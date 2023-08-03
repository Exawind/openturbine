#pragma once

#include "src/rigid_pendulum_poc/state.h"
#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

/*! Calculates the residual vector for the heavy top problem from Brüls and Cardona (2010)
 *  "On the use of Lie group time integrators in multibody dynamics," 2010, Journal of
 *  Computational and Nonlinear Dynamics, Vol 5.
 *  Ref: https://doi.org/10.1115/1.4001370
 */
HostView1D heavy_top_residual_vector(
    const HostView1D, const HostView1D, const HostView1D, const HostView1D
);

/// Calculates the iteration matrix for the heavy top problem
HostView2D heavy_top_iteration_matrix(
    const double&, const double&, const HostView1D, const HostView1D, const HostView1D,
    const double& = 1., const HostView1D = create_vector({1., 1., 1.})
);

/// Calculates the generalized coordinates residual vector for the heavy top problem
HostView1D heavy_top_gen_coords_residual_vector(
    const HostView2D, const HostView2D, const HostView1D, const HostView1D, const HostView1D,
    const HostView1D reference_position_vector
);

/// Calculates the constraint residual vector for the heavy top problem
HostView1D heavy_top_constraints_residual_vector(
    const HostView2D, const HostView1D, const HostView1D reference_position_vector
);

/// Calculates the constraint gradient matrix for the heavy top problem
HostView2D heavy_top_constraint_gradient_matrix(
    const HostView2D, const HostView1D reference_position_vector
);

/// Calculates the tangent damping matrix for the heavy top problem
HostView2D heavy_top_tangent_damping_matrix(const HostView1D, const HostView2D);

/// Calculates the tangent stiffness matrix for the heavy top problem
HostView2D heavy_top_tangent_stiffness_matrix(
    const HostView2D, const HostView1D, const HostView1D reference_position_vector
);

HostView2D heavy_top_tangent_operator(const HostView1D psi);

// TODO: Move this to its own source files when implemented
HostView1D rigid_pendulum_residual_vector(size_t size);
HostView2D rigid_pendulum_iteration_matrix(size_t size);

}  // namespace openturbine::rigid_pendulum
