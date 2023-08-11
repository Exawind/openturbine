#pragma once

#include <Kokkos_Core.hpp>

#include "src/utilities/openturbine_types.h"

namespace openturbine::gen_alpha_solver {

using openturbine::util::HostView1D;
using openturbine::util::HostView2D;

/// @brief Solve a linear system of equations using LAPACKE's dgesv
/// @details This function solves a linear system of equations using LAPACKE's
///     dgesv function. The system is of the form Ax = b, where A is a square matrix
///     (n x n) of coefficients, x is a vector (n x 1) of unknowns, and b is a vector
///     (n x 1) of right hand side values. The solution is stored in the vector b.
/// @param system An (n x n) matrix of coefficients
/// @param solution An (n x 1) vector of right hand side values
void solve_linear_system(HostView2D, HostView1D);

}  // namespace openturbine::gen_alpha_solver
