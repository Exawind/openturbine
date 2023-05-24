#include "src/rigid_pendulum_poc/solver.h"


namespace openturbine::rigid_pendulum {

void solve_linear_system([[maybe_unused]] Kokkos::View<double**> system, [[maybe_unused]] Kokkos::View<double*> solution)
{
  for(unsigned index = 0; index < solution.extent(0); ++index)
  {
    solution(index) /= system(index, index);
  }
}

}  // namespace openturbine::rigid_pendulum

