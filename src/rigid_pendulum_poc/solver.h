#pragma once

#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

/// @brief Solve a linear system of equations using LAPACKE's dgesv
/// @details This function solves a linear system of equations using LAPACKE's
///     dgesv function. The system is of the form Ax = b, where A is a square matrix
///     (nxn) of coefficients, x is a vector (nx1) of unknowns, and b is a vector
///     (nx1) of right-hand side values. The solution is stored in the vector b.
/// @param system A matrix of coefficients
/// @param solution A vector of right-hand side values
void solve_linear_system(HostView2D, HostView1D);

}  // namespace openturbine::rigid_pendulum
