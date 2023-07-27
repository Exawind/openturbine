#pragma once

#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

HostView2D heavy_top_iteration_matrix(
    HostView2D, HostView2D, HostView2D, HostView1D, HostView1D, HostView1D, const double,
    const double
);
HostView2D heavy_top_tangent_damping_matrix(HostView1D, HostView2D);
HostView2D heavy_top_tangent_stiffness_matrix(HostView1D, HostView2D, HostView1D);
HostView2D heavy_top_constraint_gradient_matrix(HostView1D, HostView2D);

HostView2D rigid_pendulum_iteration_matrix(size_t size);

}  // namespace openturbine::rigid_pendulum
