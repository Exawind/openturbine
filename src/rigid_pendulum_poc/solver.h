#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

/// @brief Solve a linear system of equations using LAPACKE's dgesv
/// @param system A matrix of coefficients
/// @param solution A vector of right-hand side values
/// @return An integer indicating the success of the solution
[[nodiscard]] int solve_linear_system(Kokkos::View<double**>, Kokkos::View<double*>);

}  // namespace openturbine::rigid_pendulum
