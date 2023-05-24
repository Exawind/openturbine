#pragma once

#include<Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

void solve_linear_system(Kokkos::View<double**> system, Kokkos::View<double*> solution);

}  // namespace openturbine::rigid_pendulum

