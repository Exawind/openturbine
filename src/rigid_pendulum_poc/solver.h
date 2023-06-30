#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

/// @brief Solve a linear system of equations using LAPACKE's dgesv
/// @details This function solves a linear system of equations using LAPACKE's
///     dgesv function. The system is of the form Ax = b, where A is a square matrix
///     (nxn) of coefficients, x is a vector (nx1) of unknowns, and b is a vector
///     (nx1) of right-hand side values. The solution is stored in the vector b.
/// @param system A matrix of coefficients
/// @param solution A vector of right-hand side values
void
    solve_linear_system(Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace>, Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>);

}  // namespace openturbine::rigid_pendulum
