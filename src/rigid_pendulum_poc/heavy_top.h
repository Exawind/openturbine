#pragma once

#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

/// Calculates the residual vector for the heavy top problem from Brüls and Cardona (2010)
/// Ref: https://doi.org/10.1115/1.4001370
HostView1D heavy_top_residual_vector(
    HostView2D, HostView2D, HostView1D, HostView1D, HostView1D, HostView1D
);

/// Calculates the iteration matrix for the heavy top problem
HostView2D heavy_top_iteration_matrix(
    HostView2D, HostView2D, HostView2D, HostView1D, HostView1D, HostView1D, const double,
    const double
);

/// Calculates the tangent stiffness matrix for the heavy top problem
HostView2D heavy_top_constraint_gradient_matrix(HostView1D, HostView2D);

/// Calculates the tangent damping matrix for the heavy top problem
HostView2D heavy_top_tangent_damping_matrix(HostView1D, HostView2D);

/// Calculates the tangent stiffness matrix for the heavy top problem
HostView2D heavy_top_tangent_stiffness_matrix(HostView1D, HostView2D, HostView1D);

// TODO: Move this to its own source files
HostView2D rigid_pendulum_iteration_matrix(size_t size);

}  // namespace openturbine::rigid_pendulum
