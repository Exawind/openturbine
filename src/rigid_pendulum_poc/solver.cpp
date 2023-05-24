#include <lapacke.h>

#include "src/rigid_pendulum_poc/solver.h"


namespace openturbine::rigid_pendulum {

int solve_linear_system(Kokkos::View<double**> system, Kokkos::View<double*> solution)
{
  int rows = system.extent(0);
  int rightHandSides = 1;
  int leadingDimension = rows;

  auto pivots = Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace>("pivots", solution.size());

  return LAPACKE_dgesv(LAPACK_ROW_MAJOR, rows, rightHandSides, system.data(), leadingDimension, pivots.data(), solution.data(), 1);
}

}  // namespace openturbine::rigid_pendulum

